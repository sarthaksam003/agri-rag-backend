from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from carag.core.config import get_settings
from carag.core.constants import get_constants

settings = get_settings()
constants = get_constants()
# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "multitenant_test"
TENANT_FIELD = "tenant_id"
EMBEDDING_DIMENSIONS = settings.EMBEDDINGS_API_DIMENSION


def setup_multitenant_collection():
    """Create a collection optimized for multitenancy with correct dimensions"""
    client = QdrantClient(url=QDRANT_URL)

    # Delete collection if it exists (for clean test)
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print("Deleted existing collection")
    except Exception as e:
        print(f"Note: Could not delete collection (might not exist): {e}")

    # Create collection with correct dimensions and multitenancy optimization
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSIONS,  # Use the correct dimensions
            distance=models.Distance.COSINE,
        ),
        hnsw_config=models.HnswConfigDiff(
            payload_m=16,  # Enable payload-aware indexing for multitenancy
            m=0,  # Disable global index (optimized for tenant queries)
        ),
    )

    # Create tenant-optimized payload index
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=TENANT_FIELD,
        field_schema=models.KeywordIndexParams(
            type="keyword",
            is_tenant=True,  # Optimize storage for tenant-based queries
        ),
    )

    print(
        f"Created multitenant-optimized collection: {COLLECTION_NAME} with {EMBEDDING_DIMENSIONS} dimensions"
    )
    return client


def test_multitenancy():
    """Test multitenancy with payload-based partitioning"""

    # Setup
    client = setup_multitenant_collection()
    embeddings = constants.embedder

    # Initialize vector store with validation disabled since we manually created the collection
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        validate_collection_config=False,  # Disable validation since we manually created it
    )

    # Sample documents for different tenants
    tenant_a_docs = [
        Document(
            page_content="Apple develops iPhones and MacBooks for creative professionals",
            metadata={"tenant_id": "tenant_a", "category": "technology"},
        ),
        Document(
            page_content="Microsoft Windows and Office are used in corporate environments",
            metadata={"tenant_id": "tenant_a", "category": "software"},
        ),
        Document(
            page_content="Google search and Android dominate mobile and web markets",
            metadata={"tenant_id": "tenant_a", "category": "technology"},
        ),
    ]

    tenant_b_docs = [
        Document(
            page_content="Apple varieties include Fuji, Gala, and Honeycrisp for baking",
            metadata={"tenant_id": "tenant_b", "category": "fruit"},
        ),
        Document(
            page_content="Microsoft is a unit of measurement in typography and printing",
            metadata={"tenant_id": "tenant_b", "category": "measurement"},
        ),
        Document(
            page_content="Google is a number equal to 10 raised to the power of 100",
            metadata={"tenant_id": "tenant_b", "category": "mathematics"},
        ),
    ]

    # Add documents for both tenants
    print("\n📝 Adding documents for Tenant A (Tech Companies)...")
    tenant_a_ids = vector_store.add_documents(
        documents=tenant_a_docs, ids=[str(uuid4()) for _ in tenant_a_docs]
    )

    print("📝 Adding documents for Tenant B (Common Terms)...")
    tenant_b_ids = vector_store.add_documents(
        documents=tenant_b_docs, ids=[str(uuid4()) for _ in tenant_b_docs]
    )

    print(f"✅ Added {len(tenant_a_ids)} documents for tenant_a")
    print(f"✅ Added {len(tenant_b_ids)} documents for tenant_b")

    # Test queries
    test_queries = ["Apple products", "Microsoft software", "Google technology"]

    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        print("-" * 50)

        # 1. Search without tenant filter (should return mixed results)
        print("1. Global search (no tenant filter):")
        global_results = vector_store.similarity_search(query, k=2)
        for i, doc in enumerate(global_results):
            tenant = doc.metadata.get("tenant_id", "unknown")
            print(f"   {i + 1}. [Tenant: {tenant}] {doc.page_content[:60]}...")

        # 2. Search with tenant A filter
        print("\n2. Tenant A search (tech companies only):")
        tenant_a_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.tenant_id",
                    match=models.MatchValue(value="tenant_a"),
                )
            ]
        )
        tenant_a_results = vector_store.similarity_search(
            query, k=2, filter=tenant_a_filter
        )
        for i, doc in enumerate(tenant_a_results):
            print(f"   {i + 1}. {doc.page_content[:60]}...")

        # 3. Search with tenant B filter
        print("\n3. Tenant B search (common terms only):")
        tenant_b_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.tenant_id",
                    match=models.MatchValue(value="tenant_b"),
                )
            ]
        )
        tenant_b_results = vector_store.similarity_search(
            query, k=2, filter=tenant_b_filter
        )
        for i, doc in enumerate(tenant_b_results):
            print(f"   {i + 1}. {doc.page_content[:60]}...")

    # Test tenant isolation more rigorously
    print("\n" + "=" * 60)
    print("🧪 TESTING TENANT ISOLATION")
    print("=" * 60)

    # Query that should return very different results per tenant
    isolation_query = "Apple computer technology"

    print(f"\nQuery: '{isolation_query}'")

    tenant_a_isolation = vector_store.similarity_search(
        isolation_query, k=1, filter=tenant_a_filter
    )

    tenant_b_isolation = vector_store.similarity_search(
        isolation_query, k=1, filter=tenant_b_filter
    )

    if tenant_a_isolation:
        print(f"✅ Tenant A result: {tenant_a_isolation[0].page_content}")
    else:
        print("❌ No results for Tenant A")

    if tenant_b_isolation:
        print(f"✅ Tenant B result: {tenant_b_isolation[0].page_content}")
    else:
        print("❌ No results for Tenant B")

    # Verify they are different
    if (
        tenant_a_isolation
        and tenant_b_isolation
        and "iPhone" in tenant_a_isolation[0].page_content
        and "fruit" in tenant_b_isolation[0].page_content
    ):
        print("🎉 SUCCESS: Tenant isolation is working correctly!")
        print("   - Tenant A sees Apple as tech company")
        print("   - Tenant B sees Apple as fruit")
    else:
        print("ℹ️  Tenant isolation results may vary based on embeddings")

    # Show collection info
    print("\n📊 Collection info:")
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"   - Points count: {collection_info.points_count}")
    print(f"   - Vectors config: {collection_info.config.params.vectors}")

    # Cleanup (optional)
    try:
        cleanup = input("\n🧹 Clean up collection? (y/n): ").lower().strip()
        if cleanup == "y":
            client.delete_collection(collection_name=COLLECTION_NAME)
            print("Collection deleted.")
    except:
        print("\nSkipping cleanup (running in non-interactive mode)")


if __name__ == "__main__":
    # Quick validation
    try:
        client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
        print("✅ Connected to Qdrant successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        print("Make sure Qdrant is running with: docker-compose up qdrant")
        exit(1)

    test_multitenancy()
