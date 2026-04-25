import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from modules.ingestion.loader import load_file
from modules.processing.splitter import split_docs
from modules.embedding.embedder import get_embedder
from modules.vectorstore.faiss_store import create_vectorstore
from modules.vectorstore.retriever import get_retriever
from modules.rag.llm import get_llm
from modules.rag.citation import render_citations
from modules.rag.pipeline import ask_question, ask_question_corag
from modules.rag.reranker import get_reranker
from logs.logs import log_rag_steps

load_dotenv()

def is_unknown_answer(answer_text):
    if not answer_text:
        return True
    answer = answer_text.strip().lower()
    unknown_patterns = [
        "i don't know because this information is not in the document.",
        "toi khong biet vi thong tin nay khong co trong tai lieu.",
        "khong tim thay thong tin trong tai lieu",
    ]
    return any(pattern in answer for pattern in unknown_patterns)


def split_combined_answer(content):
    """Extract RAG/CoRAG answer text from combined markdown content."""
    rag_answer = ""
    corag_answer = ""

    if not content:
        return rag_answer, corag_answer

    rag_marker = "### RAG\n"
    corag_marker = "\n\n### CoRAG\n"

    if rag_marker in content and corag_marker in content:
        rag_start = content.find(rag_marker) + len(rag_marker)
        corag_start = content.find(corag_marker)
        rag_answer = content[rag_start:corag_start].strip()
        corag_answer = content[corag_start + len(corag_marker):].strip()

    return rag_answer, corag_answer


def render_answer_section(title, answer, docs, query, evaluation=None, show_evaluation=False):
    st.markdown(f"### {title}")
    st.markdown(answer)

    if docs and not is_unknown_answer(answer):
        render_citations(f"Nguồn trích dẫn {title}", docs, query)

    if show_evaluation and evaluation:
        with st.expander(f"📊 Chi tiết đánh giá {title}", expanded=False):
            st.write(
                f"**Số lần thử:** {evaluation.get('attempts', 1)} | "
                f"**Điểm tin cậy:** {evaluation.get('score', 0)}/10 | "
                f"**Confidence:** {evaluation.get('confidence_score', 0)}%"
            )
            if evaluation.get("multi_hop_steps"):
                st.write("**Multi-hop reasoning steps:**")
                for i, step in enumerate(evaluation.get("multi_hop_steps", []), 1):
                    st.write(f"- {step}")
            if evaluation.get("reason"):
                st.write(f"**Lý do đánh giá:** {evaluation.get('reason')}")


TOP_K_RETRIEVE = 35
TOP_K_RERANK = 10
MIN_RERANK_SCORE = 0.22

st.set_page_config(page_title="SmartDoc AI", page_icon="📄")

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f2f6 0%, #ffffff 100%);
    }
    .sidebar-title {
        font-size: 18px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 15px;
    }
    .sidebar-section {
        margin: 15px 0;
        padding: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📄 SmartDoc AI")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "reset_uploader" not in st.session_state:
    st.session_state.reset_uploader = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "reranker" not in st.session_state:
    st.session_state.reranker = None
if "confirm_clear_history" not in st.session_state:
    st.session_state.confirm_clear_history = False
if "confirm_clear_vector" not in st.session_state:
    st.session_state.confirm_clear_vector = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "available_sources" not in st.session_state:
    st.session_state.available_sources = []
if "available_types" not in st.session_state:
    st.session_state.available_types = []
if "available_dates" not in st.session_state:
    st.session_state.available_dates = []
if "corag_enabled" not in st.session_state:
    st.session_state.corag_enabled = False  

# Always keep Self-RAG enabled.
st.session_state.self_rag_enabled = True


if st.session_state.reset_uploader:
    st.session_state.uploader_key += 1
    st.session_state.reset_uploader = False

with st.sidebar:    
    st.markdown('<div class="sidebar-title">📖 Hỗ Trợ & Cấu Hình</div>', unsafe_allow_html=True)
    
    with st.expander("❓ Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        **Cách sử dụng:**
        1. Tải tài liệu lên (PDF, DOCX)
        2. Ứng dụng sẽ xử lý tài liệu
        3. Đặt câu hỏi về tài liệu
        4. AI sẽ trả lời dựa trên nội dung
        """)
    
    with st.expander("⚙️ Cấu hình mô hình", expanded=False):
        model_options = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
        ]
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = model_options[0]

        selected_model = st.selectbox(
            "Chọn model",
            model_options,
            index=model_options.index(st.session_state.selected_model),
        )
        st.session_state.selected_model = selected_model
    
    with st.expander("🎨 Cấu hình Chunk Strategy", expanded=False):
        chunk_size = st.slider("Chọn Chunk Size", 200, 4000, 1100, 100)
        chunk_overlap = st.slider("Chọn Chunk Overlap", 0, 500, 220, 10)
        if chunk_overlap >= chunk_size:
            st.warning("Overlap phải nhỏ hơn Chunk Size.")

    with st.expander("🔧 Cấu hình CoRAG", expanded=False):
        st.session_state.corag_enabled = st.checkbox(
            "🔄 Bật CoRAG (Corrective RAG)",
            value=st.session_state.corag_enabled,
            help="CoRAG sửa lỗi truy xuất và cải thiện câu trả lời."
        )
    st.divider()

    st.markdown('<div class="sidebar-title">📄 SmartDoc AI</div>', unsafe_allow_html=True)
    st.markdown("**Tải tài liệu**")
    uploaded_files = st.file_uploader(
        "Chọn tài liệu",
        type=["pdf", "docx"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
        accept_multiple_files=True,
    )

    if st.button("➕ Chat Mới", use_container_width=True):
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.reset_uploader = True
        st.rerun()
    
    if st.button("🧹 Clear History", use_container_width=True):
        st.session_state.confirm_clear_history = True

    if st.button("🧽 Clear Vector Store", use_container_width=True):
        st.session_state.confirm_clear_vector = True

    if st.session_state.confirm_clear_history:
        st.warning("Bạn chắc chắn muốn xóa lịch sử chat?")
        col1, col2 = st.columns(2)
        if col1.button("Yes", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.session_state.confirm_clear_history = False
            st.rerun()
        if col2.button("No", use_container_width=True):
            st.session_state.confirm_clear_history = False

    if st.session_state.confirm_clear_vector:
        st.warning("Bạn chắc chắn muốn xóa vector store và file đã upload?")
        col1, col2 = st.columns(2)
        if col1.button("Yes", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.reset_uploader = True
            st.session_state.confirm_clear_vector = False
            st.rerun()
        if col2.button("No", use_container_width=True):
            st.session_state.confirm_clear_vector = False

    with st.expander("🕘 Lịch sử chat", expanded=False):
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.write("Me:", chat.get("question", ""))
                answer_text = chat.get("answer", "")
                if answer_text:
                    st.caption(answer_text)
                st.write("---")
        else:
            st.markdown("Chưa có lịch sử chat.")

# ====================== HIỂN THỊ LỊCH SỬ TIN NHẮN ======================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            saved_query = message.get("query", "")
            rag_docs = message.get("rag_docs", [])
            corag_docs = message.get("corag_docs", [])

            rag_answer = message.get("rag_answer", "")
            corag_answer = message.get("corag_answer", "")

            if not rag_answer and not corag_answer:
                rag_answer, corag_answer = split_combined_answer(message.get("content", ""))

            render_answer_section(
                "RAG",
                rag_answer,
                rag_docs,
                saved_query,
                evaluation=message.get("rag_eval"),
                show_evaluation=st.session_state.self_rag_enabled,
            )

            if st.session_state.corag_enabled:
                render_answer_section(
                    "CoRAG",
                    corag_answer,
                    corag_docs,
                    saved_query,
                    evaluation=message.get("corag_eval"),
                    show_evaluation=True,
                )
        else:
            st.markdown(message["content"])

st.divider()

query = st.chat_input("Nhập câu hỏi tại đây...")

if query:
    if st.session_state.retriever is None:
        st.error("Vui lòng upload tài liệu trước!")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            llm = get_llm(st.session_state.selected_model)
            if st.session_state.reranker is None:
                st.session_state.reranker = get_reranker()

            # RAG với Self-Evaluation
            rag_answer, rag_docs, rag_eval = ask_question(
                query,
                st.session_state.retriever,
                llm,
                chat_history=st.session_state.chat_history,
                reranker=st.session_state.reranker,
                top_k_retrieve=TOP_K_RETRIEVE,
                top_k_rerank=TOP_K_RERANK,
                min_rerank_score=MIN_RERANK_SCORE,
                self_rag_enabled=st.session_state.self_rag_enabled,
                return_evaluation=True,
            )

            # CoRAG
            if st.session_state.corag_enabled:
                corag_answer, corag_docs, corag_eval = ask_question_corag(
                    query,
                    st.session_state.retriever,
                    llm,
                    chat_history=st.session_state.chat_history,
                    reranker=st.session_state.reranker,
                    top_k_retrieve=TOP_K_RETRIEVE,
                    top_k_rerank=TOP_K_RERANK,
                    min_rerank_score=MIN_RERANK_SCORE,
                    self_rag_enabled=st.session_state.self_rag_enabled,
                    return_evaluation=True,
                )
            else:
                # CoRAG bị tắt - gán giá trị mặc định
                corag_answer = "CoRAG đã bị tắt trong cấu hình"
                corag_docs = []
                corag_eval = {
                    "score": 0,
                    "reason": "CoRAG disabled",
                    "is_sufficient": False,
                    "attempts": 0,
                    "confidence": 0.0,
                    "confidence_score": 0,
                }

            render_answer_section(
                "RAG",
                rag_answer,
                rag_docs,
                query,
                evaluation=rag_eval,
                show_evaluation=st.session_state.self_rag_enabled,
            )

            if st.session_state.corag_enabled:
                render_answer_section(
                    "CoRAG",
                    corag_answer,
                    corag_docs,
                    query,
                    evaluation=corag_eval,
                    show_evaluation=True,
                )
                combined_answer = f"### RAG\n{rag_answer}\n\n### CoRAG\n{corag_answer}"
            else:
                combined_answer = f"### RAG\n{rag_answer}"

            # Lưu vào session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": combined_answer,
                "query": query,
                "rag_answer": rag_answer,
                "corag_answer": corag_answer if st.session_state.corag_enabled else "",
                "rag_docs": rag_docs,
                "corag_docs": corag_docs if st.session_state.corag_enabled else [],
                "rag_eval": rag_eval,
                "corag_eval": corag_eval if st.session_state.corag_enabled else {},
            })
            st.session_state.chat_history.append({
                "question": query,
                "answer": combined_answer,
                "rag_answer": rag_answer,
                "corag_answer": corag_answer if st.session_state.corag_enabled else "",
            })

# ====================== PHẦN UPLOAD TÀI LIỆU ======================
if uploaded_files and st.session_state.retriever is None:
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    all_docs = []
    all_chunks = []
    sources = []
    types = []
    dates = []
    upload_date = datetime.now().strftime("%Y-%m-%d")

    with st.status("Tài liệu đã được tải lên thành công! Đang xử lý tài liệu của bạn.", expanded=True) as status:
        st.write("Đang xử lý tài liệu...")
        for uploaded_file in uploaded_files:
            file_path = os.path.join(raw_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            docs = load_file(file_path)
            file_type = os.path.splitext(uploaded_file.name)[1].lstrip(".")
            for doc in docs:
                doc.metadata.setdefault("source", uploaded_file.name)
                doc.metadata["doc_type"] = file_type
                doc.metadata["upload_date"] = upload_date

            if docs and docs[0].metadata.get("has_images"):
                st.info("Phát hiện tài liệu có hình ảnh - có thể mất nhiều thời gian hơn để xử lý.")

            st.write(f"Chia tài liệu {uploaded_file.name} thành nhiều chunks...")
            chunks = split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            log_rag_steps(documents=chunks)

            all_docs.extend(docs)
            all_chunks.extend(chunks)
            sources.append(uploaded_file.name)
            types.append(file_type)
            dates.append(upload_date)

        num_pages = len(all_docs)
        num_chunks = len(all_chunks)
        num_characters = sum(len(doc.page_content or "") for doc in all_docs)
        st.write(
            f"✅ Đang xử lý tài liệu: {num_pages} trang - {num_chunks} chunks - {num_characters} ký tự"
        )

        st.write("Tạo embedding và cơ sở dữ liệu vector...")
        embedder = get_embedder()
        vectorstore = create_vectorstore(all_chunks, embedder)

        st.write(f"✅ Cơ sở dữ liệu vector đã được tạo với {num_chunks} embeddings!")
        st.session_state.retriever = get_retriever(
            vectorstore,
            all_chunks,
            k=TOP_K_RETRIEVE,
            bm25_k=TOP_K_RETRIEVE,
        )
        st.session_state.available_sources = sorted(set(sources))
        st.session_state.available_types = sorted(set(types))
        st.session_state.available_dates = sorted(set(dates))
        status.update(label="Xử lý hoàn tất", state="complete", expanded=False)