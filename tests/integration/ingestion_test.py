import os

import pytest
from langchain_qdrant import RetrievalMode

from carag.core.config import get_settings
from carag.core.constants import get_constants
from carag.services.document_ingestion.ingest import (
    DocumentIngestionEngine,
)
from carag.services.document_ingestion.models import DocumentChunk, IngestedDocument

settings = get_settings()
constants = get_constants()

# ==============================================================================
# MODULE-LEVEL FIXTURES (Available to all test classes in this file)
# ==============================================================================


@pytest.fixture(scope="module")
def sample_pdf_paths():
    """Return paths to the test PDF files. Module scope is efficient."""
    return ["data/1.pdf", "data/2.pdf"]


@pytest.fixture
def test_tenant_id():
    """Return a unique test tenant ID for each test function."""
    return "test-tenant-123"


@pytest.fixture
def test_collection_name():
    """Use a dedicated test collection name to avoid conflicts."""
    return "test_cadocs"


@pytest.fixture
def ingestion_engine(test_tenant_id, test_collection_name):
    """
    Return a configured DocumentIngestionEngine instance with a clean test collection.
    """
    # Delete the test collection if it exists to start fresh
    try:
        collections = constants.vectorstore.get_collections().collections
        collection_names = [col.name for col in collections]
        if test_collection_name in collection_names:
            constants.vectorstore.delete_collection(
                collection_name=test_collection_name
            )
    except Exception as e:
        print(f"Warning: Could not delete test collection: {e}")

    # Create engine with test collection name
    engine = DocumentIngestionEngine(
        tenant_id=test_tenant_id, collection_name=test_collection_name
    )

    yield engine

    # Clean up after test
    try:
        engine.delete_tenant_data()
        constants.vectorstore.delete_collection(collection_name=test_collection_name)
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")


# ==============================================================================
# TEST CLASSES
# ==============================================================================


class TestDocumentIngestionEngine:
    """Test suite for DocumentIngestionEngine with real PDFs and actual API calls."""

    def test_engine_initialization(
        self, ingestion_engine, test_tenant_id, test_collection_name
    ):
        """Test that the engine initializes correctly."""
        assert ingestion_engine.tenant_id == test_tenant_id
        assert ingestion_engine.collection_name == test_collection_name
        assert isinstance(ingestion_engine.collection_name, str)

    def test_ensure_collection_exists(self, ingestion_engine):
        """Test that collection creation works."""
        ingestion_engine._ensure_collection_exists()
        # The test passes if no exception is raised.

    def test_pdf_to_markdown_conversion(self, ingestion_engine, sample_pdf_paths):
        """Test PDF to markdown conversion with actual PDF files."""
        pdf_path = sample_pdf_paths[0]
        if os.path.exists(pdf_path):
            markdown_content = ingestion_engine._pdf_to_markdown(pdf_path)
            assert isinstance(markdown_content, str)
            assert len(markdown_content.strip()) > 100

    def test_chunk_markdown_content(self, ingestion_engine):
        """Test markdown chunking functionality."""
        sample_markdown = "# Title\n\nContent paragraph."
        metadata = {"source": "test.pdf"}
        chunks = ingestion_engine._chunk_markdown(sample_markdown, metadata)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.metadata["tenant_id"] == ingestion_engine.tenant_id

    def test_embed_chunks(self, ingestion_engine):
        """Test embedding functionality with hybrid embeddings."""
        sample_chunks = [
            DocumentChunk(
                content="This is a test chunk for hybrid embeddings.",
                metadata={"tenant_id": ingestion_engine.tenant_id, "chunk_index": 0},
                id=f"{ingestion_engine.tenant_id}_chunk_0",
            ),
            DocumentChunk(
                content="Another test chunk with different content.",
                metadata={"tenant_id": ingestion_engine.tenant_id, "chunk_index": 1},
                id=f"{ingestion_engine.tenant_id}_chunk_1",
            ),
        ]
        embedded_chunks = ingestion_engine._embed_chunks(sample_chunks)
        assert isinstance(embedded_chunks, list)
        assert len(embedded_chunks) == len(sample_chunks)

        # Test that both dense and sparse embeddings are generated
        for chunk in embedded_chunks:
            assert isinstance(chunk.embedding, list)  # Dense embedding
            assert len(chunk.embedding) > 0
            assert hasattr(chunk, "sparse_embedding")  # Sparse embedding
            assert chunk.sparse_embedding is not None
            # Sparse embeddings should have indices and values
            assert hasattr(chunk.sparse_embedding, "indices")
            assert hasattr(chunk.sparse_embedding, "values")
            assert len(chunk.sparse_embedding.indices) > 0
            assert len(chunk.sparse_embedding.values) > 0

    def test_ingest_single_document_integration(
        self, ingestion_engine, sample_pdf_paths
    ):
        """Test complete document ingestion pipeline with hybrid embeddings."""
        pdf_path = sample_pdf_paths[0]
        if not os.path.exists(pdf_path):
            pytest.skip("Test PDF file not found.")

        ingested_doc = ingestion_engine.ingest_single_document(file_path=pdf_path)

        assert isinstance(ingested_doc, IngestedDocument)
        assert len(ingested_doc.chunks) > 0
        assert ingested_doc.metadata["tenant_id"] == ingestion_engine.tenant_id

        # Verify hybrid embeddings are present in chunks
        for chunk in ingested_doc.chunks:
            assert chunk.embedding is not None  # Dense embedding
            assert chunk.sparse_embedding is not None  # Sparse embedding

    def test_ingest_multiple_documents(self, ingestion_engine, sample_pdf_paths):
        """Test multiple document ingestion with hybrid embeddings."""
        existing_pdfs = [path for path in sample_pdf_paths if os.path.exists(path)]
        if len(existing_pdfs) < 2:
            pytest.skip("Need at least 2 PDF files for this test")

        results = ingestion_engine.ingest_multiple_documents(file_paths=existing_pdfs)

        assert isinstance(results, list)
        assert len(results) == len(existing_pdfs)

        # Verify hybrid embeddings in all ingested documents
        for result in results:
            for chunk in result.chunks:
                assert chunk.embedding is not None
                assert chunk.sparse_embedding is not None

    def test_search_within_tenant(self, ingestion_engine, sample_pdf_paths):
        """Test search functionality within tenant context using hybrid search."""
        pdf_path = sample_pdf_paths[0]
        if not os.path.exists(pdf_path):
            pytest.skip("No test PDF files found")

        ingestion_engine.ingest_single_document(pdf_path)

        # Test hybrid search (default)
        results = ingestion_engine.search_within_tenant(
            query="convolutional neural networks", k=1
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_with_different_retrieval_modes(
        self, ingestion_engine, sample_pdf_paths
    ):
        """Test search with different retrieval modes (DENSE, SPARSE, HYBRID)."""
        pdf_path = sample_pdf_paths[0]
        if not os.path.exists(pdf_path):
            pytest.skip("No test PDF files found")

        ingestion_engine.ingest_single_document(pdf_path)

        query = "convolutional neural networks"

        # Test dense-only search
        dense_results = ingestion_engine.search_within_tenant(
            query=query, k=1, retrieval_mode=RetrievalMode.DENSE
        )
        assert isinstance(dense_results, list)

        # Test sparse-only search
        sparse_results = ingestion_engine.search_within_tenant(
            query=query, k=1, retrieval_mode=RetrievalMode.SPARSE
        )
        assert isinstance(sparse_results, list)

        # Test hybrid search (explicit)
        hybrid_results = ingestion_engine.search_within_tenant(
            query=query, k=1, retrieval_mode=RetrievalMode.HYBRID
        )
        assert isinstance(hybrid_results, list)

    def test_search_with_score(self, ingestion_engine, sample_pdf_paths):
        """Test search functionality that returns scores."""
        pdf_path = sample_pdf_paths[0]
        if not os.path.exists(pdf_path):
            pytest.skip("No test PDF files found")

        ingestion_engine.ingest_single_document(pdf_path)

        # Test search with scores
        results_with_scores = ingestion_engine.search_with_score(
            query="convolutional neural networks", k=1
        )

        assert isinstance(results_with_scores, list)
        if len(results_with_scores) > 0:
            for doc, score in results_with_scores:
                assert hasattr(doc, "page_content")
                assert hasattr(doc, "metadata")
                assert isinstance(score, float)
                assert 0 <= score <= 1  # Scores should be normalized

    def test_tenant_isolation(self, sample_pdf_paths, test_collection_name):
        """Test that different tenants have data isolation."""
        pdf_path = sample_pdf_paths[0]
        if not os.path.exists(pdf_path):
            pytest.skip("No test PDF files found")

        tenant1_engine = DocumentIngestionEngine(
            tenant_id="tenant-iso-1", collection_name=test_collection_name
        )
        tenant2_engine = DocumentIngestionEngine(
            tenant_id="tenant-iso-2", collection_name=test_collection_name
        )

        tenant1_engine.delete_tenant_data()
        tenant2_engine.delete_tenant_data()

        tenant1_engine.ingest_single_document(pdf_path)

        count1_before = tenant1_engine.get_tenant_document_count()
        count2_before = tenant2_engine.get_tenant_document_count()

        assert count1_before > 0
        assert count2_before == 0

        # Clean up
        tenant1_engine.delete_tenant_data()
        tenant2_engine.delete_tenant_data()

    def test_delete_operations(self, ingestion_engine, sample_pdf_paths):
        """Test data deletion operations."""
        pdf_path = sample_pdf_paths[0]
        if not os.path.exists(pdf_path):
            pytest.skip("No test PDF files found")

        ingestion_engine.ingest_single_document(pdf_path)
        count_before_delete = ingestion_engine.get_tenant_document_count()
        assert count_before_delete > 0

        source_filename = os.path.basename(pdf_path)
        delete_result = ingestion_engine.delete_documents_by_source(source_filename)
        assert delete_result is True

        count_after_delete = ingestion_engine.get_tenant_document_count()
        assert count_after_delete == 0

    def test_hybrid_collection_info(self, ingestion_engine, sample_pdf_paths):
        """Test that collection info includes hybrid embedding details."""
        pdf_path = sample_pdf_paths[0]
        if not os.path.exists(pdf_path):
            pytest.skip("No test PDF files found")

        ingestion_engine.ingest_single_document(pdf_path)

        collection_info = ingestion_engine.get_collection_info()
        assert isinstance(collection_info, dict)
        assert collection_info["collection_name"] == ingestion_engine.collection_name
        assert collection_info["tenant_id"] == ingestion_engine.tenant_id
        assert collection_info["tenant_vectors_count"] > 0


class TestDocumentIngestionEdgeCases:
    """Test edge cases and error scenarios."""

    def test_non_existent_file_handling(self, ingestion_engine):
        """Test error handling for a non-existent file."""
        with pytest.raises(FileNotFoundError):
            ingestion_engine.ingest_single_document("non_existent_file.pdf")

    def test_empty_content_handling(self, ingestion_engine):
        """Test handling of empty markdown content."""
        sample_markdown = ""  # Empty content
        metadata = {"source": "empty.pdf"}

        chunks = ingestion_engine._chunk_markdown(sample_markdown, metadata)
        assert isinstance(chunks, list)
        # Should handle empty content gracefully

    def test_embedding_empty_chunks(self, ingestion_engine):
        """Test embedding functionality with empty chunks."""
        sample_chunks = [
            DocumentChunk(
                content="",  # Empty content
                metadata={"tenant_id": ingestion_engine.tenant_id, "chunk_index": 0},
                id=f"{ingestion_engine.tenant_id}_empty_chunk",
            )
        ]

        # This should handle empty content without crashing
        embedded_chunks = ingestion_engine._embed_chunks(sample_chunks)
        assert isinstance(embedded_chunks, list)


# ==============================================================================
# PYTEST CONFIGURATION (Remains at the end of the file)
# ==============================================================================


@pytest.fixture(scope="session", autouse=True)
def check_test_pdfs():
    """Check if test PDFs exist before running tests."""
    required_pdfs = ["data/1.pdf", "data/2.pdf"]
    if not all(os.path.exists(p) for p in required_pdfs):
        pytest.exit(
            "Missing test PDF files in data/ directory. Aborting tests.", returncode=1
        )


def pytest_configure(config):
    """Pytest configuration hook."""
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(items):
    """Mark all tests in this file as integration tests."""
    for item in items:
        if "ingestion_test" in item.nodeid:
            item.add_marker(pytest.mark.integration)
