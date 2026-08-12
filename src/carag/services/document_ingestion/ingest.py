import hashlib
import os
import sys
import time
import traceback
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document

# from langchain_qdrant import Qdrant
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from loguru import logger
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    KeywordIndexParams,
    MatchValue,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from carag.core.config import get_settings
from carag.core.constants import get_constants
from carag.services.document_ingestion.chunker import get_chunker
from carag.services.document_ingestion.models import DocumentChunk, IngestedDocument

settings = get_settings()
constants = get_constants()
chunker = get_chunker()

QDRANT_URL = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
TENANT_FIELD = "tenant_id"


class DocumentIngestionEngine:
    def __init__(self, tenant_id: str, collection_name: str | None = None):
        self.tenant_id: str = tenant_id
        self.collection_name: str = collection_name or settings.QDRANT_COLLECTION_NAME
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Ensure the Qdrant collection exists with proper multi-tenant configuration."""
        try:
            collections = constants.vectorstore.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                # Create collection with multi-tenancy optimization
                constants.vectorstore.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=settings.EMBEDDINGS_API_DIMENSION,
                            distance=Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        )
                    },
                    hnsw_config=HnswConfigDiff(
                        payload_m=16,  # Enable payload-aware indexing for multitenancy
                        m=0,  # Disable global index (optimized for tenant queries)
                    ),
                )

                # Create tenant-optimized payload index
                constants.vectorstore.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=TENANT_FIELD,
                    field_schema=KeywordIndexParams(
                        type="keyword",
                        is_tenant=True,  # Optimize storage for tenant-based queries
                    ),
                )
                logger.info(
                    f"Created new multi-tenant optimized collection: {self.collection_name}"
                )
            else:
                logger.info(f"Using existing collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def _pdf_to_markdown(self, file_path: str) -> str:
        """Convert PDF to markdown using MineruLoader."""
        try:
            loader = constants.loader(file_path)
            documents = loader.load()

            # Combine all document pages into a single markdown string
            markdown_content = "\n\n".join([doc.page_content for doc in documents])
            return markdown_content

        except Exception as e:
            logger.error(f"Error converting PDF to markdown for {file_path}: {e}")
            raise

    def _file_already_ingested(self, file_path: str) -> bool:
        """True → same bytes already stored for this tenant."""
        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        dup_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.tenant_id", match=MatchValue(value=self.tenant_id)
                ),
                FieldCondition(
                    key="metadata.file_sha256", match=MatchValue(value=file_hash)
                ),
            ]
        )
        count = constants.vectorstore.count(
            collection_name=self.collection_name, count_filter=dup_filter
        ).count
        return count > 0

    def _chunk_markdown(
        self, markdown_content: str, metadata: dict[str, Any]
    ) -> list[DocumentChunk]:
        """Chunk markdown content using the custom chunker."""
        try:
            chunks = chunker.chunk(markdown_content, metadata)

            document_chunks: list[DocumentChunk] = []
            for i, chunk in enumerate(chunks):
                # Create document chunk with metadata including tenant_id
                chunk_metadata = metadata.copy()
                chunk_metadata.update(
                    {
                        "tenant_id": self.tenant_id,  # Add tenant_id to metadata
                        "chunk_index": i,
                        "chunk_id": f"chunk_{i}",
                        "total_chunks": len(chunks),
                    }
                )

                document_chunk = DocumentChunk(
                    content=chunk, metadata=chunk_metadata, id=str(uuid4())
                )
                document_chunks.append(document_chunk)

            return document_chunks

        except Exception as e:
            logger.error(f"Error chunking markdown content: {e}")
            raise

    def _embed_chunks(
        self, document_chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """Embed document chunks using the embedder."""
        try:
            # Extract texts for batch embedding
            texts = [chunk.content for chunk in document_chunks]

            # Batch embed all texts
            dense_embeddings = constants.embedder.embed_documents(texts)
            sparse_embeddings = constants.sparse_embedder.embed_documents(texts)

            # Assign embeddings to chunks
            for chunk, dense_emb, sparse_emb in zip(
                document_chunks, dense_embeddings, sparse_embeddings
            ):
                chunk.embedding = dense_emb  # Dense embedding
                chunk.sparse_embedding = sparse_emb

            return document_chunks

        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            raise

    def _store_in_qdrant(
        self,
        ingested_document: IngestedDocument,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> None:
        """Store document chunks in Qdrant vector store with tenant isolation."""
        try:
            # Convert to LangChain documents for Qdrant integration
            langchain_documents: list[Document] = []
            metadatas: list[dict[str, Any]] = []
            ids: list[str] = []

            for chunk in ingested_document.chunks:
                if chunk.embedding is None or chunk.sparse_embedding is None:
                    logger.warning(
                        f"Warning: Empty embeddings recieved for the chunk:\n{chunk}"
                    )
                    continue

                # Ensure tenant_id is in metadata
                if "tenant_id" not in chunk.metadata:
                    chunk.metadata["tenant_id"] = self.tenant_id

                # Create LangChain document
                langchain_doc = Document(
                    page_content=chunk.content, metadata=chunk.metadata
                )
                langchain_documents.append(langchain_doc)
                metadatas.append(chunk.metadata)
                ids.append(chunk.id)

            # Use LangChain's Qdrant integration for storage with tenant isolation
            vector_store = QdrantVectorStore(
                client=constants.vectorstore,
                collection_name=self.collection_name,
                embedding=constants.embedder,
                sparse_embedding=constants.sparse_embedder,
                retrieval_mode=retrieval_mode,
                vector_name="dense",
                sparse_vector_name="sparse",
            )
            # Add documents to Qdrant
            vector_store.add_documents(documents=langchain_documents, ids=ids)

            logger.info(
                f"Stored {len(langchain_documents)} chunks in Qdrant for tenant {self.tenant_id}"
            )

        except Exception as e:
            logger.error(f"Error storing in Qdrant: {e}")
            raise

    def create_tenant_filter(self) -> Filter:
        """Create a filter for the current tenant."""
        return Filter(
            must=[
                FieldCondition(
                    key="metadata.tenant_id",
                    match=MatchValue(value=self.tenant_id),
                )
            ]
        )

    def search_within_tenant(
        self,
        query: str,
        k: int = 4,
        additional_filters: Filter | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> list[Document]:
        """Search for documents within the current tenant's data."""
        try:
            # Create tenant filter
            tenant_filter = self.create_tenant_filter()

            # Combine with additional filters if provided
            if additional_filters:
                final_filter = Filter(must=[tenant_filter, additional_filters])
            else:
                final_filter = tenant_filter

            # Initialize Qdrant with validation disabled for multi-tenant setup
            qdrant = QdrantVectorStore(
                client=constants.vectorstore,
                collection_name=self.collection_name,
                embedding=constants.embedder,
                sparse_embedding=constants.sparse_embedder,
                retrieval_mode=retrieval_mode,
                vector_name="dense",
                sparse_vector_name="sparse",
            )

            # Perform similarity search with tenant filter
            results = qdrant.similarity_search(query=query, k=k, filter=final_filter)

            return results

        except Exception as e:
            logger.error(f"Error searching within tenant {self.tenant_id}: {e}")
            raise

    def search_with_score(
        self,
        query: str,
        k: int = 4,
        additional_filters: Filter | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> list[tuple[Document, float]]:
        """Search for documents and return scores using hybrid search."""
        try:
            # Create tenant filter
            tenant_filter = self.create_tenant_filter()

            # Combine with additional filters if provided
            if additional_filters:
                final_filter = Filter(must=[tenant_filter, additional_filters])
            else:
                final_filter = tenant_filter

            # Initialize Qdrant with hybrid retrieval mode
            vector_store = QdrantVectorStore(
                client=constants.vectorstore,
                collection_name=self.collection_name,
                embedding=constants.embedder,
                sparse_embedding=constants.sparse_embedder,
                retrieval_mode=retrieval_mode,
                vector_name="dense",
                sparse_vector_name="sparse",
            )

            # Perform hybrid similarity search with scores
            results = vector_store.similarity_search_with_score(
                query=query, k=k, filter=final_filter
            )

            return results

        except Exception as e:
            logger.error(f"Error searching with score for tenant {self.tenant_id}: {e}")
            raise

    def get_tenant_document_count(self) -> int:
        """Get the number of documents for the current tenant."""
        try:
            tenant_filter = self.create_tenant_filter()

            # Count points for this tenant
            count_result = constants.vectorstore.count(
                collection_name=self.collection_name, count_filter=tenant_filter
            )

            return count_result.count

        except Exception as e:
            logger.error(
                f"Error getting document count for tenant {self.tenant_id}: {e}"
            )
            return 0

    def list_documents(self) -> list[dict[str, str]]:
        """
        Return every distinct document that has been ingested for the current tenant.
        Each item is a dict with keys:
            id   – the document_id stored in the chunk metadata
            filename – the original filename stored in the chunk metadata
        """
        try:
            seen_ids: set[str] = set()
            docs: list[dict[str, str]] = []

            # scroll through all points that belong to this tenant
            next_offset = None
            while True:
                page, next_offset = constants.vectorstore.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=self.create_tenant_filter(),
                    limit=1_000,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )

                for rec in page:
                    meta = rec.payload.get("metadata", {})
                    doc_id = meta.get("document_id")
                    name = meta.get("original_filename")
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        docs.append({"id": doc_id, "filename": name or "unknown"})

                if next_offset is None:  # Qdrant signals end of collection
                    break

            return docs

        except Exception as e:
            logger.error(f"Error listing documents for tenant {self.tenant_id}: {e}")
            return []

    def ingest_single_document(
        self,
        file_path: str,
        additional_metadata: dict[str, Any] | None = None,
    ) -> IngestedDocument:
        """
        Ingest a single PDF document for the current tenant.
        Duplicate bytes (per tenant) are rejected.
        """
        start_time = time.time()

        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Document not found: {file_path}")

            # ---- 1.  duplicate detection ---------------------------------
            if self._file_already_ingested(file_path):
                logger.warning(
                    f"Skipping duplicate PDF (tenant {self.tenant_id}): {file_path}"
                )
                raise FileExistsError(
                    "This PDF has already been ingested for the tenant."
                )

            # compute hash once and reuse
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            # ---- 2.  prepare metadata ----------------------------------
            file_stats = os.stat(file_path)
            filename = os.path.basename(file_path)

            base_metadata = {
                "tenant_id": self.tenant_id,
                "source": filename,
                "file_path": file_path,
                "file_size": file_stats.st_size,
                "file_modified": file_stats.st_mtime,
                "file_sha256": file_hash,  # duplicate guard
            }
            if additional_metadata:
                base_metadata.update(additional_metadata)

            logger.info(
                f"Starting ingestion of: {filename} for tenant: {self.tenant_id}"
            )

            # ---- 3.  pdf → markdown → chunks → embeddings → qdrant -------
            markdown_content = self._pdf_to_markdown(file_path)
            chunks = self._chunk_markdown(markdown_content, base_metadata)
            embedded_chunks = self._embed_chunks(chunks)

            ingested_doc = IngestedDocument(
                original_filename=filename,
                file_path=file_path,
                file_size=file_stats.st_size,
                chunks=embedded_chunks,
                metadata=base_metadata,
                processing_time=time.time() - start_time,
            )

            self._store_in_qdrant(ingested_doc)

            logger.info(
                f"Successfully ingested document in {ingested_doc.processing_time:.2f}s "
                f"for tenant {self.tenant_id}"
            )
            return ingested_doc

        except Exception as e:
            logger.error(
                f"Failed to ingest document {file_path} for tenant {self.tenant_id}: {e}"
            )
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_text = "".join(tb_lines)
            logger.error(tb_text)
            raise e

    def ingest_multiple_documents(
        self,
        file_paths: list[str],
        additional_metadata: dict[str, Any] | None = None,
    ) -> list[IngestedDocument]:
        """Ingest multiple PDF documents for the current tenant."""
        results = []

        for file_path in file_paths:
            try:
                result = self.ingest_single_document(file_path, additional_metadata)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed to ingest {file_path} for tenant {self.tenant_id}: {e}"
                )
                # Continue with other documents even if one fails
                continue

        return results

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the current collection."""
        try:
            collection_info = constants.vectorstore.get_collection(self.collection_name)

            # Get tenant-specific count
            tenant_count = self.get_tenant_document_count()

            return {
                "collection_name": self.collection_name,
                "tenant_id": self.tenant_id,
                "total_vectors_count": collection_info.vectors_count,
                "total_points_count": collection_info.points_count,
                "tenant_vectors_count": tenant_count,
                "status": collection_info.status,
                "is_multi_tenant_optimized": True,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def delete_tenant_data(self) -> bool:
        """Delete all data for the current tenant."""
        try:
            tenant_filter = self.create_tenant_filter()

            constants.vectorstore.delete(
                collection_name=self.collection_name, points_selector=tenant_filter
            )

            logger.info(f"Deleted all data for tenant {self.tenant_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting data for tenant {self.tenant_id}: {e}")
            return False

    def delete_document_chunks(self, chunk_ids: list[str]) -> bool:
        """Delete specific chunks by their IDs."""
        try:
            constants.vectorstore.delete(
                collection_name=self.collection_name, points_selector=chunk_ids
            )
            logger.info(f"Deleted {len(chunk_ids)} specific chunks")
            return True
        except Exception as e:
            logger.error(f"Error deleting specific chunks: {e}")
            return False

    def delete_documents_by_document_id(self, document_id: str) -> tuple[bool, int]:
        """Delete all chunks for a specific document_id. Returns (success, deleted_count)."""
        try:
            # First count how many documents we're going to delete
            document_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.tenant_id",
                        match=MatchValue(value=self.tenant_id),
                    ),
                    FieldCondition(
                        key="metadata.document_id", match=MatchValue(value=document_id)
                    ),
                ]
            )

            existing_count = constants.vectorstore.count(
                collection_name=self.collection_name, count_filter=document_filter
            ).count

            if existing_count == 0:
                logger.warning(f"No documents found with document_id: {document_id}")
                return False, 0

            delete_result = constants.vectorstore.delete(
                collection_name=self.collection_name, points_selector=document_filter
            )

            logger.info(
                f"Deleted {existing_count} chunks for document_id: {document_id}"
            )
            return True, existing_count

        except Exception as e:
            logger.error(f"Error deleting documents by document_id {document_id}: {e}")
            return False, 0


def get_ingestion_engine(
    tenant_id: str, collection_name: str | None = None
) -> DocumentIngestionEngine:
    """Get a configured ingestion engine instance for a specific tenant."""
    return DocumentIngestionEngine(tenant_id=tenant_id, collection_name=collection_name)
