import unittest
from types import SimpleNamespace

from modules.rag.pipeline import ask_question, ask_question_cog


class FakeDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}
    

class FakeRetriever:
    def __init__(self, docs=None):
        self.docs = docs or []
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.docs

    def get_relevant_documents(self, query):
        self.queries.append(query)
        return self.docs


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        prompt_lower = prompt.lower()

        if "standalone question" in prompt_lower or "câu hỏi độc lập" in prompt_lower:
            return SimpleNamespace(content="What are the main findings and their implications?")

        if "generate" in prompt_lower or "tạo" in prompt_lower:
            return SimpleNamespace(content="Main findings\nImplications")

        if (
            "answer only using the provided context" in prompt_lower
            or "you are a document-grounded assistant" in prompt_lower
            or "trả lời dựa trên ngữ cảnh" in prompt_lower
        ):
            if "installation procedure" in prompt_lower:
                return SimpleNamespace(
                    content=(
                        "1. Download the installer.\n"
                        "2. Run the setup file.\n"
                        "3. Follow the on-screen instructions."
                    )
                )
            if "main findings" in prompt_lower and "implications" in prompt_lower:
                return SimpleNamespace(
                    content=(
                        "The paper finds that the new method improves accuracy.\n"
                        "Its implication is that the approach is suitable for larger deployments."
                    )
                )
            if "differential equations" in prompt_lower:
                return SimpleNamespace(
                    content="I don't know because this information is not in the document."
                )

        if "answer only using context" in prompt_lower or "ngữ cảnh" in prompt_lower:
            return SimpleNamespace(content="I don't know because this information is not in the document.")

        return SimpleNamespace(content="I don't know because this information is not in the document.")


class TestRagPipeline(unittest.TestCase):
    def test_simple_factual_question_returns_step_by_step_instructions(self):
        docs = [
            FakeDoc(
                "Installation procedure:\n1. Download the installer.\n2. Run the setup file.\n3. Follow the on-screen instructions.",
                {"source": "manual.pdf", "page": 1, "chunk_id": 1},
            )
        ]
        retriever = FakeRetriever(docs)
        llm = FakeLLM()

        answer, cited_docs = ask_question(
            "What is the installation procedure?",
            retriever,
            llm,
            chat_history=[],
            reranker=None,
        )

        self.assertIn("1. Download the installer.", answer)
        self.assertIn("2. Run the setup file.", answer)
        self.assertIn("3. Follow the on-screen instructions.", answer)
        self.assertEqual(cited_docs, docs)

    def test_complex_reasoning_question_returns_summary_with_analysis(self):
        docs = [
            FakeDoc(
                "The study found that the new method improves accuracy by 12%.\n"
                "This implies the approach is effective for larger-scale deployments.",
                {"source": "paper.pdf", "page": 2, "chunk_id": 1},
            )
        ]
        retriever = FakeRetriever(docs)
        llm = FakeLLM()

        answer, cited_docs = ask_question_cog(
            "What are the main findings and their implications?",
            retriever,
            llm,
            chat_history=[],
            reranker=None,
        )

        self.assertIn("improves accuracy", answer)
        self.assertIn("suitable for larger deployments", answer)
        self.assertEqual(cited_docs, docs)

    def test_out_of_context_question_returns_unknown_response(self):
        retriever = FakeRetriever([])
        llm = FakeLLM()

        answer, cited_docs = ask_question(
            "How to solve differential equations?",
            retriever,
            llm,
            chat_history=[],
            reranker=None,
        )

        self.assertEqual(cited_docs, [])
        self.assertEqual(
            answer,
            "I don't know because this information is not in the document.",
        )

    def test_return_evaluation_includes_confidence_metadata(self):
        docs = [
            FakeDoc(
                "Installation procedure:\n1. Download the installer.\n2. Run the setup file.",
                {"source": "manual.pdf", "page": 1, "chunk_id": 1},
            )
        ]
        retriever = FakeRetriever(docs)
        llm = FakeLLM()

        answer, cited_docs, evaluation = ask_question(
            "What is the installation procedure?",
            retriever,
            llm,
            chat_history=[],
            reranker=None,
            return_evaluation=True,
        )

        self.assertIn("Download the installer", answer)
        self.assertEqual(cited_docs, docs)
        self.assertIn("confidence", evaluation)
        self.assertIn("confidence_score", evaluation)
        self.assertIn("attempts", evaluation)
        self.assertGreaterEqual(evaluation["confidence_score"], 0)


if __name__ == "__main__":
    unittest.main()
