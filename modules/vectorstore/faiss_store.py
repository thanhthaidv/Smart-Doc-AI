from langchain_community.vectorstores import FAISS

def create_vectorstore(chunks, embedder):
    """Tạo kho lưu trữ vector FAISS từ các đoạn tài liệu."""
    return FAISS.from_documents(chunks, embedder)
