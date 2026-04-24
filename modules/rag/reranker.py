from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Re-rank documents with a cross-encoder model."""

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
        batch_size=16,
    ):
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        scored_docs = []
        for doc, score in zip(docs, scores):
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["rerank_score"] = float(score)
            scored_docs.append((doc, float(score)))

        scored_docs.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _score in scored_docs[:top_k]]


def get_reranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu",
    batch_size=16,
):
    """Create a cross-encoder reranker instance."""
    return CrossEncoderReranker(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
