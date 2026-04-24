import os
from datetime import datetime
import re
from html import escape

import streamlit as st
from dotenv import load_dotenv

from modules.ingestion.loader import load_file
from modules.processing.splitter import split_docs
from modules.embedding.embedder import get_embedder
from modules.vectorstore.faiss_store import create_vectorstore
from modules.vectorstore.retriever import get_retriever
from modules.rag.llm import get_llm
from modules.rag.pipeline import ask_question
from modules.rag.reranker import get_reranker
from logs.logs import log_rag_steps

load_dotenv()

# =========================
# HIGHLIGHT KEYWORDS
# =========================
def highlight_keywords_in_text(text, query):
    """
    Tô vàng các từ khóa từ query trong text.
    
    Args:
        text: Nội dung chunk
        query: Câu hỏi/từ khóa cần tìm
    
    Returns:
        HTML string với keywords được highlight
    """
    if not query or not text:
        return escape(text)
    
    keywords = query.lower().split()
    text_escaped = escape(text)
        
    for keyword in keywords:
        if len(keyword) > 2:  # Chỉ highlight từ có > 2 ký tự
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            text_escaped = pattern.sub(
                f'<mark style="background-color: #FFD700; padding: 2px 4px; border-radius: 3px;"><b>{keyword}</b></mark>',
                text_escaped
            )
    
    return text_escaped


TOP_K_RETRIEVE = 30  
TOP_K_RERANK = 15    
MIN_RERANK_SCORE = 0.1  

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
if "source_name" not in st.session_state:
    st.session_state.source_name = None
if "show_full_history" not in st.session_state:
    st.session_state.show_full_history = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "confirm_clear_history" not in st.session_state:
    st.session_state.confirm_clear_history = False
if "confirm_clear_vector" not in st.session_state:
    st.session_state.confirm_clear_vector = False
if "available_sources" not in st.session_state:
    st.session_state.available_sources = []
if "available_types" not in st.session_state:
    st.session_state.available_types = []
if "available_dates" not in st.session_state:
    st.session_state.available_dates = []
if "reranker" not in st.session_state:
    st.session_state.reranker = None

if st.session_state.reset_uploader:
    st.session_state.uploader_key += 1
    st.session_state.reset_uploader = False

with st.sidebar:    
    st.markdown('<div class="sidebar-title">📖 Hỗ Trợ & Cấu Hình</div>', unsafe_allow_html=True)
    
    with st.expander("❓ Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        **Cách sử dụng:**
        1. Tải tài liệu lên (PDF, DOCX, XLSX, PPTX)
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
        chunk_size = st.slider(
            "Chọn Chunk Size (Kích thước đoạn)",
            min_value=200,
            max_value=4000,
            value=1000,
            step=100,
        )
        chunk_overlap = st.slider(
            "Chọn Chunk Overlap (Độ gối đầu)",
            min_value=0,
            max_value=500,
            value=100,
            step=10,
        )
        if chunk_overlap >= chunk_size:
            st.warning("Overlap phải nhỏ hơn Chunk Size.")

    st.divider()

    

    st.markdown('<div class="sidebar-title">📄 SmartDoc AI</div>', unsafe_allow_html=True)
    st.markdown("**Tải tài liệu**")
    uploaded_files = st.file_uploader(
        "Chọn tài liệu",
        type=["pdf", "docx", "xlsx", "pptx"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
        accept_multiple_files=True,
    )



    if  st.button("➕ Chat Mới", use_container_width=True):
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
                #st.write("AI:", chat.get("answer", ""))
                st.write("---")
        else:
            st.markdown("Chưa có lịch sử chat.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
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
            answer, cited_docs = ask_question(
                query,
                st.session_state.retriever,
                llm,
                chat_history=st.session_state.chat_history,
                reranker=st.session_state.reranker,
                top_k_retrieve=TOP_K_RETRIEVE,
                top_k_rerank=TOP_K_RERANK,
                min_rerank_score=MIN_RERANK_SCORE,
            )
            st.markdown(answer)
            if cited_docs:
                st.markdown("**Nguồn trích dẫn:**")
                for idx, doc in enumerate(cited_docs, start=1):
                    source = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", "N/A")
                    doc_type = doc.metadata.get("doc_type", "unknown")
                    upload_date = doc.metadata.get("upload_date", "N/A")
                    title = (
                        f"[{idx}] {os.path.basename(source)} - Trang {page} - "
                        f"{doc_type} - {upload_date}"
                    )
                    with st.expander(title, expanded=False):
                        # Highlight từ khóa từ câu hỏi
                        highlighted_text = highlight_keywords_in_text(
                            doc.page_content,
                            query
                        )
                        highlight_html = (
                            "<div style='background:#f8f9fa; padding:10px; border-radius:6px; border-left: 3px solid #FFD700;'>"
                            f"{highlighted_text}"
                            "</div>"
                        )
                        st.markdown(highlight_html, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append({"question": query, "answer": answer})

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