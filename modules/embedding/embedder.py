from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedder():
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    model_kwargs = {"device": "cpu"}

    encode_kwargs = {"normalize_embeddings": True}

    """Lấy thể hiện của mô hình nhúng."""
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)