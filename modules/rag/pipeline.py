import json
import re
from typing import Dict, List, Tuple

VI_HINTS = [
    " la ", " va ", " cua ", " trong ", " tai lieu ", " khong ", " duoc ", " nhu the nao ",
]
VI_CHARS = set("áșŁáșąĂŁĂáșĄáș ÄÄáșŻáș°áșłáșČáș”áșŽáș·áș¶ĂąĂáș„áș€áș§áșŠáș©áșšáș«áșȘáș­áșŹÄÄĂ©ĂĂšĂáș»áșșáșœáșŒáșčáșžĂȘĂáșżáșŸá»á»á»á»á»á»á»á»Ă­ĂĂŹĂá»á»Ä©Äšá»á»ĂłĂĂČĂá»á»Ă”Ăá»á»ĂŽĂá»á»á»á»á»á»á»á»á»á»ÆĄÆ á»á»á»á»á»á»á»Ąá» á»Łá»ąĂșĂĂčĂá»§á»ŠĆ©Ćšá»„á»€Æ°ÆŻá»©á»šá»«á»Șáș­á»Źá»Żá»źá»±á»°ĂœĂá»łá»Čá»·á»¶á»čá»žá»”á»Ž")
HAN_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+")
UNKNOWN_VI = "Toi khong biet vi thong tin nay khong co trong tai lieu."
UNKNOWN_EN = "I don't know because this information is not in the document."


def _extract_text(response) -> str:
    if response is None:
        return ""
    if hasattr(response, "content"):
        return (response.content or "").strip()
    return str(response).strip()


def _normalize_text(text: str) -> str:
    return (text or "").strip().replace("```json", "").replace("```", "").strip()


def _parse_json_response(raw: str) -> Dict:
    cleaned = _normalize_text(raw)
    if not cleaned:
        return {}

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


def _contains_han(text: str) -> bool:
    return bool(HAN_PATTERN.search(text or ""))


def _strip_han(text: str) -> str:
    cleaned = HAN_PATTERN.sub("", text or "")
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _detect_language(question: str) -> str:
    q = f" {(question or '').lower()} "
    if any(hint in q for hint in VI_HINTS):
        return "vi"
    if any(ch in question for ch in VI_CHARS):
        return "vi"
    return "en"


def _unknown_answer(lang: str) -> str:
    return UNKNOWN_VI if lang == "vi" else UNKNOWN_EN


def _language_instruction(lang: str) -> str:
    if lang == "vi":
        return (
            "Tra loi 100% bang tieng Viet. "
            "Khong duoc dung ky tu Han. "
            "Neu tai lieu khong co thong tin, tra loi dung 1 cau: "
            '"Toi khong biet vi thong tin nay khong co trong tai lieu."'
        )
    return (
        "Answer only in English. "
        "If the document does not contain enough evidence, answer exactly: "
        '"I don\'t know because this information is not in the document."'
    )


def _format_history(chat_history: List[Dict] = None, max_turns: int = 4) -> str:
    if not chat_history:
        return ""

    lines = []
    recent_turns = chat_history[-max_turns:]
    for turn in recent_turns:
        question = (turn.get("question") or "").strip()
        answer = (turn.get("answer") or "").strip()
        if question:
            lines.append(f"User: {question}")
        if answer:
            lines.append(f"Assistant: {answer}")

    return "\n".join(lines)


def _safe_get_docs(retriever, query, k=None):
    if hasattr(retriever, "invoke"):
        try:
            if k is not None:
                return retriever.invoke(query, k=k)
            return retriever.invoke(query)
        except TypeError:
            docs = retriever.invoke(query)
            return docs[:k] if k is not None else docs

    if hasattr(retriever, "get_relevant_documents"):
        try:
            if k is not None:
                return retriever.get_relevant_documents(query, k=k)
            return retriever.get_relevant_documents(query)
        except TypeError:
            docs = retriever.get_relevant_documents(query)
            return docs[:k] if k is not None else docs

    return []


def _doc_key(doc):
    meta = doc.metadata or {}
    key = (meta.get("source"), meta.get("page"), meta.get("chunk_id"))
    if any(p is not None for p in key):
        return key
    return hash((doc.page_content or "")[:300])


def _dedupe_docs(docs: List) -> List:
    unique = []
    seen = set()
    for doc in docs:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _lexical_overlap_score(question: str, content: str) -> float:
    q_words = set(_tokenize(question))
    c_words = set(_tokenize(content))
    if not q_words or not c_words:
        return 0.0
    inter = q_words & c_words
    return float(len(inter)) / float(len(q_words))


def format_docs(docs, max_chars=2800):
    if not docs:
        return "Khong co tai lieu lien quan."

    rows = []
    total = 0
    max_per_doc = max(300, max_chars // max(1, len(docs)))

    for idx, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        page = meta.get("page", "N/A")
        score = meta.get("rerank_score", 0.0)
        content = (doc.page_content or "").strip()
        if len(content) > max_per_doc:
            content = content[:max_per_doc] + "..."

        row = f"[{idx}] (Nguon: {source}, Trang: {page}, Score: {score:.3f})\n{content}"
        rows.append(row)

        total += len(content)
        if total >= max_chars:
            break

    return "\n\n".join(rows)


# ==================== SELF-RAG: QUERY REWRITING ====================
def _query_rewriting(question: str, llm, lang: str, history_text: str = "") -> str:
    """
    Query Rewriting: Sử dụng LLM để viết lại câu hỏi follow-up hoặc câu hỏi mơ hồ
    thành phiên bản rõ ràng, dễ tìm kiếm hơn.
    """
    prompt = (
        "You are a query rewriting expert. Rewrite the user's question to make it "
        "more specific, clear, and effective for document retrieval.\n"
        "Rules:\n"
        "- Keep the original meaning and intent\n"
        "- Resolve ambiguous references using conversation history\n"
        "- Make implicit constraints explicit\n"
        "- Return ONLY the rewritten query, no explanation\n"
        f"Language: {lang}\n"
        f"Conversation history:\n{history_text or 'None'}\n"
        f"Original question: {question}\n"
        "Rewritten query:"
    )
    rewritten = _normalize_text(_extract_text(llm.invoke(prompt)))
    if not rewritten:
        return question
    rewritten = rewritten.replace("Rewritten query:", "").strip()
    if _contains_han(rewritten) and lang != "zh":
        return _strip_han(rewritten) or question
    return rewritten


# ==================== SELF-RAG: MULTI-HOP REASONING ====================
def _multi_hop_decomposition(question: str, llm, lang: str, max_steps: int = 3) -> List[str]:
    """
    Multi-hop Reasoning: Phân tích câu hỏi phức tạp thành tối đa 3 bước tìm kiếm logic.
    """
    prompt = (
        "You are a multi-hop reasoning expert. Decompose the complex question into "
        f"up to {max_steps} simple sub-questions that can be answered step by step.\n"
        "Rules:\n"
        "- Each sub-question should be self-contained and searchable\n"
        "- Order sub-questions logically\n"
        "- Return ONLY numbered list, no explanations\n"
        f"Language: {lang}\n"
        f"Question: {question}\n"
        "Sub-questions:"
    )
    raw = _normalize_text(_extract_text(llm.invoke(prompt)))
    steps = []
    for line in raw.splitlines():
        item = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
        if item and len(item) > 3:
            steps.append(item)
    return steps[:max_steps] if steps else [question]


def _multi_hop_retrieve(
    question: str,
    retriever,
    llm,
    lang: str,
    top_k_retrieve: int,
    max_steps: int = 3,
) -> Tuple[List, List[str]]:
    """
    Multi-hop Retrieval: Truy vấn lần lượt các bước và tổng hợp kết quả.
    """
    steps = _multi_hop_decomposition(question, llm, lang, max_steps=max_steps)
    hop_docs = []
    per_step_k = max(3, top_k_retrieve // 3)  # Mỗi step lấy ít hơn, tổng thể vẫn đủ

    for step in steps:
        step_docs = _safe_get_docs(retriever, step, k=per_step_k)
        hop_docs.extend(step_docs)
        # Optional: ghi log từng bước

    return _dedupe_docs(hop_docs), steps


# ==================== SELF-RAG: SELF-EVALUATION ====================
def _self_evaluate(
    question: str,
    context: str,
    answer: str,
    llm,
    lang: str,
    docs: List,
) -> Dict:
    """
    Self-Evaluation: LLM đánh giá lại câu trả lời theo thang điểm 1-10,
    đưa ra lý do và quyết định.
    """
    # Nếu answer đã là "I don't know"
    if "khong biet" in answer.lower() or "don't know" in answer.lower():
        return {
            "score": 2,
            "reason": "unknown_answer",
            "is_sufficient": False,
            "confidence": 0.05
        }
    
    # Đánh giá chi tiết
    prompt = (
        "You are a strict answer evaluator. Evaluate the answer based on:\n"
        "1. Relevance (does it answer the question?)\n"
        "2. Groundedness (is it supported by the context?)\n"
        "3. Completeness (does it cover all aspects?)\n\n"
        f"Question: {question}\n"
        f"Context (truncated): {context[:2000]}\n"
        f"Answer: {answer}\n\n"
        "Return JSON only with this format:\n"
        "{\n"
        '  "score": <1-10 integer>,\n'
        '  "reason": "<brief explanation in Vietnamese or English>",\n'
        '  "is_sufficient": <true/false>,\n'
        '  "missing_info": "<what information is missing, if any>"\n'
        "}\n"
        f"Language for reason: {lang}\n"
        "JSON:"
    )
    
    raw = _extract_text(llm.invoke(prompt))
    data = _parse_json_response(raw)
    
    if data:
        if "is_sufficient" not in data:
            data["is_sufficient"] = data.get("score", 0) >= 6
        if "score" not in data:
            data["score"] = 5 if data.get("is_sufficient") else 2
        return data
    
    # Fallback
    supported = _basic_answer_supported(question, answer, docs)
    return {
        "score": 6 if supported else 3,
        "reason": "fallback evaluation",
        "is_sufficient": supported,
        "missing_info": ""
    }


def _basic_answer_supported(question: str, answer: str, docs: List) -> bool:
    if not answer or not docs:
        return False
    if "i don't know" in answer.lower() or "toi khong biet" in answer.lower():
        return False

    joined = " ".join((doc.page_content or "")[:400] for doc in docs)
    overlap = _lexical_overlap_score(question + " " + answer, joined)
    return overlap >= 0.08


def _confidence_from_evaluation(evaluation: Dict) -> float:
    """Confidence Scoring: Kết hợp điểm từ Self-Evaluation để trả về mức độ tin cậy."""
    score = float(evaluation.get("score", 0))
    confidence = min(1.0, max(0.0, score / 10.0))
    
    if evaluation.get("is_sufficient"):
        confidence = max(confidence, 0.65)
    
    if "unknown" in evaluation.get("reason", "").lower():
        confidence = min(confidence, 0.1)
    
    return round(confidence, 2)


# ==================== SELF-RAG PIPELINE ====================
def _self_rag_pipeline(
    question: str,
    retriever,
    llm,
    chat_history: List[Dict] = None,
    reranker=None,
    top_k_retrieve: int = 30,
    top_k_rerank: int = 10,
    min_rerank_score: float = 0.22,
    max_retries: int = 2,
) -> Tuple[str, List, Dict]:
    """
    Self-RAG Pipeline đúng quy trình:
    1. Query Rewriting (nếu có history hoặc cần thiết)
    2. Multi-hop Reasoning (phân rã câu hỏi phức tạp)
    3. Retrieve + Rerank
    4. Generate answer
    5. Self-Evaluation + Confidence Scoring
    6. Retry nếu chưa đủ tốt (tối đa max_retries lần)
    """
    lang = _detect_language(question)
    language_instruction = _language_instruction(lang)
    history_text = _format_history(chat_history)
    
    best_answer = _unknown_answer(lang)
    best_docs = []
    best_confidence = -1.0
    final_evaluation = {
        "score": 0,
        "reason": "",
        "is_sufficient": False,
        "missing_info": "",
        "multi_hop_steps": [],
        "attempts": 0,
        "rewrites": 0,
        "confidence": 0.0,
        "confidence_score": 0,
    }
    
    for attempt in range(max_retries + 1):
        # Bước 1: Query Rewriting cho lần attempt > 0
        if attempt == 0:
            current_query = question
        else:
            current_query = _query_rewriting(question, llm, lang, history_text)
            final_evaluation["rewrites"] = attempt
        
        # Bước 2: Multi-hop Reasoning (chỉ thực hiện cho câu hỏi phức tạp)
        # Phát hiện câu hỏi phức tạp (có từ nối, so sánh, hoặc dài)
        is_complex = any(word in question.lower() for word in 
                        ["và", "va", "so sánh", "khác nhau", "giống nhau", "liệt kê", "trình bày"])
        
        if is_complex or attempt > 0:
            hop_docs, hop_steps = _multi_hop_retrieve(
                question=current_query,
                retriever=retriever,
                llm=llm,
                lang=lang,
                top_k_retrieve=top_k_retrieve,
            )
            final_evaluation["multi_hop_steps"] = hop_steps
        else:
            hop_docs = []
            hop_steps = []
        
        # Bước 3: Retrieve + Rerank
        retrieved_docs = _safe_get_docs(retriever, current_query, k=top_k_retrieve)
        
        # Kết hợp với multi-hop docs nếu có
        if hop_docs:
            combined_docs = _dedupe_docs(retrieved_docs + hop_docs)
        else:
            combined_docs = retrieved_docs
        
        # Rerank
        if reranker:
            reranked = _apply_rerank(question, combined_docs, reranker, top_k_rerank, min_rerank_score)
        else:
            reranked = combined_docs[:top_k_rerank]
        
        selected_docs = reranked[:7]
        
        if not selected_docs:
            if attempt < max_retries:
                continue
            else:
                break
        
        # Bước 4: Generate answer
        context = format_docs(selected_docs, max_chars=3500)
        answer = _answer_with_context_and_history(
            current_query,
            context,
            llm,
            language_instruction,
            history_text,
        )
        
        # Bước 5: Self-Evaluation
        evaluation = _self_evaluate(question, context, answer, llm, lang, selected_docs)
        evaluation["attempts"] = attempt + 1
        evaluation["confidence"] = _confidence_from_evaluation(evaluation)
        evaluation["confidence_score"] = int(round(evaluation["confidence"] * 100))
        
        # Cập nhật best
        if evaluation["confidence"] >= best_confidence:
            best_confidence = evaluation["confidence"]
            best_answer = answer
            best_docs = selected_docs
            final_evaluation.update(evaluation)
            final_evaluation["multi_hop_steps"] = hop_steps
            final_evaluation["rewrites"] = attempt
        
        # Bước 6: Dừng nếu đủ tốt
        if evaluation.get("is_sufficient", False) and evaluation.get("score", 0) >= 6:
            break
    
    # Fallback
    if "khong biet" in best_answer.lower() or "don't know" in best_answer.lower():
        final_evaluation["score"] = 2
        final_evaluation["reason"] = "Không tìm thấy thông tin đủ tin cậy"
        final_evaluation["confidence"] = min(final_evaluation.get("confidence", 0.0), 0.1)
        final_evaluation["confidence_score"] = int(round(final_evaluation["confidence"] * 100))
    
    return best_answer, best_docs, final_evaluation


# ==================== CORAG: GRADE DOCUMENTS ====================
def _grade_documents_corag(
    question: str, 
    docs: List, 
    llm, 
    lang: str, 
    batch_size: int = 5
) -> Tuple[List, List]:
    """
    CoRAG Bước 2 & 3: Grade Documents + Filter
    Trả về: (kept_docs, rejected_docs) - chỉ giữ documents được đánh giá "yes"
    """
    if not docs:
        return [], []
    
    kept = []
    rejected = []
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        contents = "\n\n---\n\n".join(
            f"Document {j+1}:\n{(doc.page_content or '')[:1200]}" 
            for j, doc in enumerate(batch)
        )
        
        prompt = (
            "You are a relevance grader. For each document, answer strictly 'yes' or 'no'.\n"
            "A document is relevant if it contains information that directly helps answer the question.\n"
            f"Language lock: {lang}\n"
            f"Question: {question}\n\n"
            f"Documents:\n{contents}\n\n"
            "Format: Document 1: yes/no\nDocument 2: yes/no\n..."
        )
        
        raw = _extract_text(llm.invoke(prompt)).lower()
        grades = re.findall(r'document\s*(\d+):\s*(yes|no)', raw)
        
        grade_map = {int(g[0]): g[1] for g in grades}
        
        for j, doc in enumerate(batch):
            doc_num = j + 1
            is_relevant = grade_map.get(doc_num, "no") == "yes"
            
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["corag_grade"] = "relevant" if is_relevant else "irrelevant"
            
            if is_relevant:
                kept.append(doc)
            else:
                rejected.append(doc)
    
    return kept, rejected


def _corag_corrective_step(
    question: str,
    llm,
    lang: str,
    history_text: str,
    rejected_docs: List,
    retriever,
    top_k_retrieve: int,
) -> Tuple[str, List]:
    """
    CoRAG Bước 4: Corrective Step
    - Viết lại câu hỏi dựa trên lý do bị reject
    - Mở rộng retrieval với query mới
    """
    if not rejected_docs:
        return question, []
    
    # Lấy mẫu nội dung bị reject để phân tích
    rejection_sample = ""
    for doc in rejected_docs[:2]:
        content_preview = (doc.page_content or "")[:300]
        rejection_sample += f"- {content_preview}...\n"
    
    prompt = (
        "The following documents were NOT relevant to the question.\n"
        "Create an improved retrieval query that will find relevant information.\n"
        f"Language: {lang}\n"
        f"Original question: {question}\n"
        f"Conversation history: {history_text or 'None'}\n"
        f"Sample of irrelevant content:\n{rejection_sample}\n"
        "Improved query (return ONLY the query text):"
    )
    
    corrective_query = _normalize_text(_extract_text(llm.invoke(prompt)))
    if not corrective_query or len(corrective_query) < 5:
        corrective_query = f"{question} chi tiết thông tin"
    
    additional_docs = _safe_get_docs(retriever, corrective_query, k=top_k_retrieve // 2)
    
    return corrective_query, additional_docs


# ==================== CORAG PIPELINE ====================
def _corag_pipeline(
    question: str,
    retriever,
    llm,
    chat_history: List[Dict] = None,
    reranker=None,
    top_k_retrieve: int = 35,
    top_k_rerank: int = 10,
    min_rerank_score: float = 0.22,
    max_corrective_attempts: int = 2,
) -> Tuple[str, List, Dict]:
    """
    CoRAG Pipeline đúng quy trình:
    Bước 1: Retrieve → Lấy documents ban đầu từ Hybrid Retriever
    Bước 2: Grade Documents → LLM đánh giá từng document (yes/no)
    Bước 3: Filter → Giữ lại chỉ những document đạt chất lượng cao
    Bước 4: Corrective Step → Nếu không đủ, viết lại câu hỏi và retrieve lại
    Bước 5: Generate → Sinh câu trả lời cuối cùng với context đã được lọc
    """
    lang = _detect_language(question)
    language_instruction = _language_instruction(lang)
    history_text = _format_history(chat_history)
    
    best_answer = _unknown_answer(lang)
    best_docs = []
    best_confidence = -1.0
    final_evaluation = {
        "score": 0,
        "reason": "",
        "is_sufficient": False,
        "missing_info": "",
        "multi_hop_steps": [],
        "attempts": 0,
        "corrections": 0,
        "kept_docs_count": 0,
        "rejected_docs_count": 0,
        "confidence": 0.0,
        "confidence_score": 0,
    }
    
    current_query = question
    all_retrieved_docs = []
    corrections_made = 0
    
    for attempt in range(max_corrective_attempts + 1):
        # Bước 1: Retrieve
        retrieved_docs = _safe_get_docs(retriever, current_query, k=top_k_retrieve)
        all_retrieved_docs.extend(retrieved_docs)
        all_retrieved_docs = _dedupe_docs(all_retrieved_docs)
        
        # Rerank nếu có
        if reranker:
            reranked = _apply_rerank(question, all_retrieved_docs, reranker, top_k_rerank, min_rerank_score)
        else:
            reranked = all_retrieved_docs[:top_k_rerank]
        
        # Bước 2 & 3: Grade Documents và Filter
        kept_docs, rejected_docs = _grade_documents_corag(
            question=question,
            docs=reranked[:top_k_retrieve],
            llm=llm,
            lang=lang,
            batch_size=5,
        )
        
        final_evaluation["kept_docs_count"] = len(kept_docs)
        final_evaluation["rejected_docs_count"] = len(rejected_docs)
        final_evaluation["attempts"] = attempt + 1
        
        # Kiểm tra xem có đủ documents tốt không (ít nhất 2 docs)
        has_sufficient_docs = len(kept_docs) >= 2
        
        if has_sufficient_docs or attempt == max_corrective_attempts:
            # Bước 5: Generate answer với documents đã được filter
            selected_docs = kept_docs[:7] if kept_docs else reranked[:5]
            
            if selected_docs:
                context = format_docs(selected_docs, max_chars=3500)
                answer = _answer_with_context_and_history(
                    question,
                    context,
                    llm,
                    language_instruction,
                    history_text,
                )
                
                evaluation = _self_evaluate(question, context, answer, llm, lang, selected_docs)
                evaluation["attempts"] = attempt + 1
                evaluation["corrections"] = corrections_made
                evaluation["confidence"] = _confidence_from_evaluation(evaluation)
                evaluation["confidence_score"] = int(round(evaluation["confidence"] * 100))
                
                if evaluation["confidence"] >= best_confidence:
                    best_confidence = evaluation["confidence"]
                    best_answer = answer
                    best_docs = selected_docs
                    final_evaluation.update(evaluation)
                    final_evaluation["corrections"] = corrections_made
                
                if evaluation.get("is_sufficient", False) and evaluation.get("score", 0) >= 6:
                    return best_answer, best_docs, final_evaluation
        
        # Bước 4: Corrective Step - nếu chưa đủ documents và còn attempts
        if not has_sufficient_docs and attempt < max_corrective_attempts:
            corrections_made += 1
            corrective_query, additional_docs = _corag_corrective_step(
                question=question,
                llm=llm,
                lang=lang,
                history_text=history_text,
                rejected_docs=rejected_docs,
                retriever=retriever,
                top_k_retrieve=top_k_retrieve,
            )
            current_query = corrective_query
            all_retrieved_docs.extend(additional_docs)
            final_evaluation["multi_hop_steps"].append(f"Correction {corrections_made}: {corrective_query}")
    
    # Fallback
    if not best_docs or "khong biet" in best_answer.lower() or "don't know" in best_answer.lower():
        final_evaluation["score"] = 2
        final_evaluation["reason"] = "Không tìm thấy thông tin đủ tin cậy sau corrective attempts"
        final_evaluation["confidence"] = min(final_evaluation.get("confidence", 0.0), 0.1)
        final_evaluation["confidence_score"] = int(round(final_evaluation["confidence"] * 100))
        final_evaluation["corrections"] = corrections_made
        if not best_answer or best_answer == "":
            best_answer = _unknown_answer(lang)
    
    return best_answer, best_docs, final_evaluation


# ==================== HELPER FUNCTIONS ====================
def _apply_rerank(
    question: str,
    docs: List,
    reranker,
    top_k_rerank: int,
    min_rerank_score: float,
) -> List:
    if not docs or not reranker:
        return docs[:top_k_rerank] if docs else []

    if hasattr(reranker, "rerank_with_deduplication"):
        ranked = reranker.rerank_with_deduplication(question, docs, top_k=top_k_rerank)
    else:
        ranked = reranker.rerank(question, docs, top_k=top_k_rerank)

    filtered = []
    for doc in ranked:
        score = (doc.metadata or {}).get("rerank_score", None)
        if score is None or score >= min_rerank_score:
            filtered.append(doc)

    return filtered if filtered else ranked[: min(3, len(ranked))]


def _answer_with_context_and_history(
    question: str,
    context: str,
    llm,
    language_instruction: str,
    history_text: str,
) -> str:
    prompt = (
        "You are a document-grounded assistant. "
        "Answer only using the provided context. "
        "Do not add external knowledge. "
        "If evidence is missing, use the required unknown sentence.\n\n"
        f"Language rule: {language_instruction}\n\n"
        f"Conversation context:\n{history_text or 'None'}\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    answer = _extract_text(llm.invoke(prompt))
    if _contains_han(answer) and "ky tu Han" in language_instruction:
        answer = _strip_han(answer)
    return answer


def _simple_pipeline(
    question: str,
    retriever,
    llm,
    top_k_retrieve: int = 30,
) -> Tuple[str, List]:
    lang = _detect_language(question)
    docs = _safe_get_docs(retriever, question, k=top_k_retrieve)
    
    if not docs:
        return _unknown_answer(lang), []

    selected_docs = _dedupe_docs(docs)[:5]
    context = format_docs(selected_docs, max_chars=2200)
    
    prompt_prefix = (
        "You are a document assistant. Read the context and answer directly. "
        "Stay grounded in the provided context only."
    )
    
    prompt = (
        f"{prompt_prefix}\n\n"
        f"Language rule: {_language_instruction(lang)}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    answer = _extract_text(llm.invoke(prompt))
    
    if not answer or (_contains_han(answer) and lang != "zh"):
        answer = _unknown_answer(lang) if not answer else _strip_han(answer)
    
    return answer, selected_docs


# ==================== PUBLIC API ====================
def ask_question(
    question: str,
    retriever,
    llm,
    chat_history: List[Dict] = None,
    reranker=None,
    top_k_retrieve: int = 30,
    top_k_rerank: int = 10,
    min_rerank_score: float = 0.22,
    self_rag_enabled: bool = True,
    return_evaluation: bool = False,
):
    """Self-RAG pipeline with full features"""
    if not self_rag_enabled:
        answer, docs = _simple_pipeline(question, retriever, llm, top_k_retrieve)
        if return_evaluation:
            return answer, docs, {"score": 5, "reason": "simple_mode", "is_sufficient": True, "confidence": 0.5, "confidence_score": 50, "attempts": 1}
        return answer, docs

    answer, docs, evaluation = _self_rag_pipeline(
        question=question,
        retriever=retriever,
        llm=llm,
        chat_history=chat_history,
        reranker=reranker,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        min_rerank_score=min_rerank_score,
        max_retries=2,
    )
    if return_evaluation:
        return answer, docs, evaluation
    return answer, docs


def ask_question_corag(   
    question: str,
    retriever,
    llm,
    chat_history: List[Dict] = None,
    reranker=None,
    top_k_retrieve: int = 35,
    top_k_rerank: int = 10,
    min_rerank_score: float = 0.22,
    self_rag_enabled: bool = True,
    return_evaluation: bool = False,
):
    """CoRAG (Corrective RAG) pipeline - Full 5-step process"""
    if not self_rag_enabled:
        answer, docs = _simple_pipeline(question, retriever, llm, top_k_retrieve)
        if return_evaluation:
            return answer, docs, {"score": 5, "reason": "simple_mode", "is_sufficient": True, "confidence": 0.5, "confidence_score": 50, "attempts": 1}
        return answer, docs
    
    answer, docs, evaluation = _corag_pipeline(
        question=question,
        retriever=retriever,
        llm=llm,
        chat_history=chat_history,
        reranker=reranker,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        min_rerank_score=min_rerank_score,
        max_corrective_attempts=2,
    )
    if return_evaluation:
        return answer, docs, evaluation
    return answer, docs


def ask_question_cog(*args, **kwargs):
    return ask_question_corag(*args, **kwargs)