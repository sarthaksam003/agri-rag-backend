from carag.services.document_ingestion.ingest import DocumentIngestionEngine


def query_documents(tenant_id: str, query: str):
    engine = DocumentIngestionEngine(tenant_id=tenant_id)
    results = engine.search_with_score(query=query, k=5)

    formatted_results = []
    for doc, score in results:
        formatted_results.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
            }
        )
    return formatted_results


if __name__ == "__main__":
    results = query_documents(
        tenant_id="demo-tenant",
        query="what is e4m3",
    )

    for i, result in enumerate(results):
        print("############################################")
        print(f"CHUNK: {i}")
        print(f"Source: {result['source']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Content: {result['content']}")  # First 200 chars
        print("---")
        print("###################################")
