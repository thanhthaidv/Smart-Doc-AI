
from langchain_community.retrievers import BM25Retriever


def _safe_get_docs(retriever, query, k=None):
    if hasattr(retriever, "invoke"):
        try:
            if k is not None:
                return retriever.invoke(query, k=k)
            return retriever.invoke(query)
        except TypeError:
            return retriever.invoke(query)
    if hasattr(retriever, "get_relevant_documents"):
        try:
            if k is not None:
                return retriever.get_relevant_documents(query, k=k)
            return retriever.get_relevant_documents(query)
        except TypeError:
            return retriever.get_relevant_documents(query)
    return []


def _doc_key(doc):
    meta = doc.metadata or {}
    key = (
        meta.get("source"),
        meta.get("page"),
        meta.get("chunk_id"),
    )
    if any(part is not None for part in key):
        return key
    return hash(doc.page_content)


class HybridRetriever:
    """Retriever kết hợp BM25 và vector search."""

    def __init__(self, bm25_retriever, vector_retriever, weights=(0.5, 0.5), rrf_k=60):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.weights = weights
        self.rrf_k = rrf_k

    def invoke(self, query, k=5):
        bm25_docs = _safe_get_docs(self.bm25_retriever, query, k=k)
        vector_docs = _safe_get_docs(self.vector_retriever, query, k=k)

        scored = {}
        for rank, doc in enumerate(bm25_docs):
            key = _doc_key(doc)
            scored.setdefault(key, {"doc": doc, "score": 0.0})
            scored[key]["score"] += self.weights[0] * (1.0 / (self.rrf_k + rank))

        for rank, doc in enumerate(vector_docs):
            key = _doc_key(doc)
            scored.setdefault(key, {"doc": doc, "score": 0.0})
            scored[key]["score"] += self.weights[1] * (1.0 / (self.rrf_k + rank))

        ranked = sorted(scored.values(), key=lambda item: item["score"], reverse=True)
        return [item["doc"] for item in ranked[:k]]


def get_retriever(vectorstore, chunks, k=5, bm25_k=5, weights=(0.5, 0.5)):
    """Tạo retriever kết hợp BM25 và vector search."""
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = bm25_k

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 30, "lambda_mult": 0.7},
    )

    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        weights=weights,
    )