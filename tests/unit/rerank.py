from langchain_core.documents import Document

from carag.core import get_constants

if __name__ == "__main__":

    def main():
        reranker = get_constants().reranker
        test_query = "What is the most important innovation in large language models?"

        test_documents = [
            Document(
                page_content="The invention of the transistor in 1947 was a pivotal moment in modern electronics.",
                metadata={"source": "history_of_tech.txt"},
            ),
            Document(
                page_content="The Transformer architecture, introduced in the paper 'Attention Is All You Need', is the foundational technology behind models like BERT and GPT.",
                metadata={"source": "llm_research_paper.pdf"},
            ),
            Document(
                page_content="Recurrent Neural Networks (RNNs) were a popular choice for sequence data but struggled with long-term dependencies.",
                metadata={"source": "old_nlp_book.md"},
            ),
            Document(
                page_content="The concept of attention mechanisms allows a model to focus on specific parts of the input sequence, which proved to be a critical breakthrough.",
                metadata={"source": "attention_mechanisms.html"},
            ),
        ]
        sync_results = reranker.compress_documents(test_documents, test_query)
        print(f"Query: '{test_query}'")
        for doc in sync_results:
            score = doc.metadata.get("relevance_score", "N/A")
            print(f"  - Score: {score:.4f} | Content: {doc.page_content}")
        print("-" * 30 + "\n")

    main()
