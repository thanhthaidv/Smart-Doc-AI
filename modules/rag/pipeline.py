from logs.logs import log_rag_steps


# =========================
# HISTORY
# =========================
def _format_history(chat_history, max_turns=6):
    if not chat_history:
        return ""
    recent = chat_history[-max_turns:]
    lines = []
    for turn in recent:
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()
        if q:
            lines.append(f"User: {q}")
        if a:
            lines.append(f"Assistant: {a}")
    return "\n".join(lines)


# =========================
# CONDENSE QUESTION
# =========================
def _condense_question(query, history_text, llm, is_vi):
    if not history_text:
        return query

    if is_vi:
        prompt = f"""
Ban la tro ly AI. Hay viet lai cau hoi moi thanh cau hoi doc lap ro nghia.

Lich su:
{history_text}

Cau hoi moi:
{query}

Cau hoi doc lap:
"""
    else:
        prompt = f"""
Rewrite the question into a standalone question.

History:
{history_text}

Question:
{query}

Standalone question:
"""

    res = llm.invoke(prompt)
    return (res.content or "").strip() or query


# =========================
# SIMPLE QUERY CHECK
# =========================
def _is_simple_query(q):
    return len(q.split()) <= 5 or len(q) < 40


# =========================
# RETRIEVE SAFE
# =========================
def _retrieve_with_k(retriever, query, k):
    try:
        return retriever.invoke(query)[:k]
    except:
        return retriever.get_relevant_documents(query)[:k]


# =========================
# RERANK FAST
# =========================
def _rerank_fast(reranker, query, docs, top_k, min_score):
    if not reranker or not docs:
        return []

    # ⚡ giảm số docs trước khi rerank
    docs = docs[:max(top_k * 3, 10)]

    pairs = [(query, d.page_content) for d in docs]

    scores = reranker.model.predict(pairs, batch_size=16)

    for doc, score in zip(docs, scores):
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["rerank_score"] = float(score)

    docs = sorted(docs, key=lambda d: d.metadata["rerank_score"], reverse=True)

    # ⚡ early stop
    results = []
    for d in docs:
        if d.metadata["rerank_score"] < min_score:
            continue
        results.append(d)
        if len(results) >= top_k:
            break

    return results


# =========================
# CONTEXT BUILD (LIMIT)
# =========================
def _build_context(docs, max_chars=3000):
    context = ""
    for d in docs:
        if len(context) > max_chars:
            break
        context += d.page_content + "\n"
    return context


# =========================
# MAIN FUNCTION
# =========================
def ask_question(
    query,
    retriever,
    llm,
    chat_history=None,
    reranker=None,
    top_k_retrieve=20,
    top_k_rerank=5,
    min_rerank_score=0.2,
):
    # detect language
    vi_chars = "ạảấầẩẫậđéèẻẽếềểễệíìỉĩịóòỏõốồổỗộúùủũứừửữự"
    is_vi = any(c in (query or "").lower() for c in vi_chars)

    # history
    history_text = _format_history(chat_history)

    # condense
    standalone_query = _condense_question(query, history_text, llm, is_vi)

    # retrieve
    retrieved_docs = _retrieve_with_k(retriever, standalone_query, top_k_retrieve)

    # =========================
    # RERANK (SMART)
    # =========================
    if reranker and not _is_simple_query(standalone_query):
        docs_for_answer = _rerank_fast(
            reranker,
            standalone_query,
            retrieved_docs,
            top_k_rerank,
            min_rerank_score,
        )

        # fallback nếu fail
        if not docs_for_answer:
            docs_for_answer = retrieved_docs[:top_k_rerank]
    else:
        docs_for_answer = retrieved_docs[:top_k_rerank]

    # =========================
    # LOG
    # =========================
    log_rag_steps(
        user_input=standalone_query,
        relevant_docs=docs_for_answer[:3],
    )

    if not docs_for_answer:
        return "Toi khong biet vi thong tin nay khong co trong tai lieu.", []

    # =========================
    # CONTEXT
    # =========================
    context = _build_context(docs_for_answer)

    # =========================
    # PROMPT
    # =========================
    if is_vi:
        prompt = f"""
Ban la tro ly AI chi tra loi dua tren NGU CANH.

Quy tac:
1. Chi duoc dung context
2. Khong biet → tra loi: "Toi khong biet vi thong tin nay khong co trong tai lieu."

Ngu canh:
{context}

Cau hoi:
{standalone_query}

Tra loi:
"""
    else:
        prompt = f"""
Answer ONLY using context.

If not found, say:
"I don't know because this information is not in the document."

Context:
{context}

Question:
{standalone_query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, docs_for_answer


