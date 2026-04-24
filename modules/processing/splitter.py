from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_docs(documents, chunk_size=1500, chunk_overlap=200):
    """Sử dụng công cụ tách văn bản để chia tài liệu thành nhiều phần."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
