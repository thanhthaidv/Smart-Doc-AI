
from langchain_community.retrievers import BM25Retriever
import numpy as np


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
    """Retriever kết hợp BM25 và vector search với normalization."""

    def __init__(self, bm25_retriever, vector_retriever, weights=(0.5, 0.5), rrf_k=60):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.weights = weights
        self.rrf_k = rrf_k

    def _get_scores_from_retriever(self, retriever, query, k):
        """Lấy documents kèm similarity scores nếu có thể."""
        docs = _safe_get_docs(retriever, query, k=k)

        # Thử lấy scores trực tiếp từ vector store
        if hasattr(retriever, "vectorstore") and hasattr(
            retriever.vectorstore, "similarity_search_with_score"
        ):
            try:
                docs_with_scores = retriever.vectorstore.similarity_search_with_score(query, k=k)
                return [(doc, score) for doc, score in docs_with_scores]
            except Exception:
                pass

        # Fallback: gán score dựa trên rank
        return [(doc, 1.0 / (self.rrf_k + idx)) for idx, doc in enumerate(docs)]

    def invoke(self, query, k=5):
        # Lấy nhiều hơn k docs để fusion hiệu quả hơn
        fetch_k = k * 3

        # Vector search với scores
        vector_results = self._get_scores_from_retriever(self.vector_retriever, query, fetch_k)
        vector_docs_scores = {_doc_key(doc): (doc, score) for doc, score in vector_results}

        # BM25 search
        bm25_docs = _safe_get_docs(self.bm25_retriever, query, k=fetch_k)

        scored = {}
        for rank, doc in enumerate(bm25_docs):
            key = _doc_key(doc)
            if key not in scored:
                scored[key] = {
                    "doc": doc,
                    "score": 0.0,
                    "bm25_rank": rank,
                    "vector_score": 0.0,
                }

            # BM25 theo RRF
            bm25_score = self.weights[0] * (1.0 / (self.rrf_k + rank))
            scored[key]["score"] += bm25_score
            scored[key]["bm25_score"] = bm25_score

        for key, (doc, sim_score) in vector_docs_scores.items():
            if key not in scored:
                scored[key] = {
                    "doc": doc,
                    "score": 0.0,
                    "bm25_rank": len(bm25_docs),
                    "vector_score": sim_score,
                }

            # Normalize similarity score về [0, 1]
            norm_sim_score = 1.0 / (1.0 + np.exp(-sim_score))
            vector_score = self.weights[1] * norm_sim_score
            scored[key]["score"] += vector_score
            scored[key]["vector_score"] = vector_score

        ranked = sorted(scored.values(), key=lambda item: item["score"], reverse=True)

        if ranked:
            print(f"Top score: {ranked[0]['score']:.4f}")

        return [item["doc"] for item in ranked[:k]]


def get_retriever(vectorstore, chunks, k=5, bm25_k=5, weights=(0.5, 0.5)):
    """Tạo retriever kết hợp BM25 và vector search."""
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = bm25_k

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k * 3},
    )

    # Lưu vectorstore để có thể lấy similarity score
    vector_retriever.vectorstore = vectorstore

    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        weights=weights,
    )