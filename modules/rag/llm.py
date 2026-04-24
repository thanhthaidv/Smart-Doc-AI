import os

from langchain_groq import ChatGroq


def get_llm(model_name: str | None = None):
    """Lấy phiên bản LLM."""
    model = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(model=model,temperature=0)
