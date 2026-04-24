import os

import pandas as pd
from docx import Document as DocxDocument
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from pptx import Presentation

from modules.ingestion.ocr import detect_pdf_has_images, detect_pdf_has_text, ocr_pdf_to_text

def load_pdf(file_path: str, use_ocr_if_needed: bool = True):
    """Tải tài liệu PDF, nhận dạng ký tự các tệp đã quét khi cần."""
    has_images = detect_pdf_has_images(file_path)

    if use_ocr_if_needed and not detect_pdf_has_text(file_path):
        text = ocr_pdf_to_text(file_path)
        return [
            Document(
                page_content=text,
                metadata={"source": file_path, "ocr": True, "has_images": has_images},
            )
        ]

    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    for idx, doc in enumerate(documents, start=1):
        doc.metadata["has_images"] = has_images
        doc.metadata.setdefault("source", file_path)
        doc.metadata.setdefault("page", doc.metadata.get("page") or doc.metadata.get("page_number"))
        doc.metadata.setdefault("chunk_id", idx)
    return documents


def load_docx(file_path: str):
    """Tải DOCX và trả về nội dung đã hợp nhất."""
    doc = DocxDocument(file_path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append("\t".join(cells))

    text = "\n".join(parts)
    return [
        Document(
            page_content=text,
            metadata={"source": file_path, "type": "docx", "chunk_id": 1},
        )
    ]


def load_xlsx(file_path: str):
    """Tải Excel và trích xuất văn bản từ tất cả các sheet."""
    sheets = pd.read_excel(file_path, sheet_name=None)
    documents = []
    for idx, (sheet_name, df) in enumerate(sheets.items(), start=1):
        text = df.to_string(index=False)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "type": "xlsx",
                    "sheet": sheet_name,
                    "chunk_id": idx,
                },
            )
        )
    return documents


def load_pptx(file_path: str):
    """Tải PPTX và trích xuất văn bản từ các slide."""
    presentation = Presentation(file_path)
    documents = []
    for idx, slide in enumerate(presentation.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text.strip())
        if texts:
            documents.append(
                Document(
                    page_content="\n".join(texts),
                    metadata={"source": file_path, "type": "pptx", "slide": idx, "chunk_id": idx},
                )
            )
    return documents


def load_file(file_path: str, use_ocr_if_needed: bool = True):
    """Điều phối để tải nhiều định dạng file."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path, use_ocr_if_needed=use_ocr_if_needed)
    if ext == ".docx":
        return load_docx(file_path)
    if ext == ".xlsx":
        return load_xlsx(file_path)
    if ext == ".pptx":
        return load_pptx(file_path)

    raise ValueError(f"Unsupported file type: {ext}")

