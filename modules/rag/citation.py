import os
from html import escape

import streamlit as st


def build_citations(retrieved_docs: list) -> list[dict]:
    """
    Tạo danh sách citation từ các docs đã retrieve.
    Mỗi citation gồm: index, page, file, snippet, source, ocr.

    Args:
        retrieved_docs: list Document từ retriever.invoke()

    Returns:
        list[dict] với keys: index, page, file, snippet, source, ocr
    """
    citations = []
    for index, doc in enumerate(retrieved_docs, start=1):
        metadata = doc.metadata or {}
        page_num = metadata.get("page", "N/A")
        raw_source = metadata.get("source", "Unknown")
        file_name = os.path.basename(raw_source) if raw_source != "Unknown" else "Unknown document"
        snippet = (doc.page_content or "")[:250].replace("\n", " ") + "..."

        citations.append(
            {
                "index": index,
                "page": page_num,
                "file": file_name,
                "snippet": snippet,
                "source": raw_source,
                "ocr": bool(metadata.get("ocr", False)),
            }
        )

    return citations


def render_citations(title: str, retrieved_docs: list, query: str) -> None:
    """Render citations in a single expander for the active answer."""
    citations = build_citations(retrieved_docs)
    if not citations:
        return

    with st.expander(title, expanded=False):
        for citation in citations:
            ocr_tag = " 🔍 (Dữ liệu từ ảnh/OCR)" if citation.get("ocr") else ""
            title_text = f"[{citation['index']}] {citation['file']} - Trang {citation['page']}{ocr_tag}"
            content = citation.get("snippet", "")
            highlighted_text = escape(content)

            st.markdown(f"**{title_text}**")
            st.markdown(
                "<div style='background:#f8f9fa; padding:10px; border-radius:6px; border-left: 3px solid #FFD700; margin-bottom: 12px;'>"
                f"{highlighted_text}"
                "</div>",
                unsafe_allow_html=True,
            )
