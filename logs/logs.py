import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _summarize_docs(docs, max_items=3):
	if not docs:
		return ""
	items = []
	for doc in docs[:max_items]:
		meta = doc.metadata or {}
		source = meta.get("source") or "unknown"
		page = meta.get("page") or meta.get("chunk_id") or "n/a"
		items.append(f"{source}:{page}")
	return ", ".join(items)


def log_rag_steps(
	documents=None,
	user_input=None,
	relevant_docs=None,
	retrieved_docs=None,
	reranked_docs=None,
	rerank_model=None,
) -> None:
	"""Ghi lại các bước xử lý RAG cơ bản."""
	if documents is not None:
		logger.info("Xử lí %s chunks", len(documents))
	if user_input is not None:
		logger.info("Query: %s", user_input)
	if relevant_docs is not None:
		logger.info("Đã truy xuất %s documents", len(relevant_docs))
	if retrieved_docs is not None:
		preview = _summarize_docs(retrieved_docs)
		logger.info("Bi-encoder/hybrid retrieved %s docs. Top: %s", len(retrieved_docs), preview)
	if reranked_docs is not None:
		preview = _summarize_docs(reranked_docs)
		label = rerank_model or "cross-encoder"
		logger.info("%s reranked to %s docs. Top: %s", label, len(reranked_docs), preview)

