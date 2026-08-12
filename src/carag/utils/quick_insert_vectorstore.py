"""
Script to insert a PDF document into the vector store
"""

import os
import sys

from carag.services.document_ingestion.ingest import (
    get_ingestion_engine,
)


def insert_pdf_document():
    """Insert a PDF document into the vector store."""

    # Configuration
    TENANT_ID = "demo-tenant"  # Change this to your tenant ID
    PDF_FILE_PATH = "data/1.pdf"  # Path to your PDF file

    # Validate file exists
    if not os.path.exists(PDF_FILE_PATH):
        print(f"❌ Error: PDF file not found at {PDF_FILE_PATH}")
        print("Please ensure the file exists and the path is correct.")
        return False

    try:
        print(f"📁 Starting PDF ingestion for tenant: {TENANT_ID}")
        print(f"📄 Processing file: {PDF_FILE_PATH}")
        print("-" * 50)

        # Initialize the ingestion engine
        engine = get_ingestion_engine(tenant_id=TENANT_ID)

        # Optional: Add custom metadata
        additional_metadata = {
            "document_type": "research_paper",  # or "manual", "article", etc.
            "category": "technology",
            "ingested_by": "script",
            "language": "en",
        }

        # Ingest the document
        print("⏳ Converting PDF to markdown...")
        ingested_doc = engine.ingest_single_document(
            file_path=PDF_FILE_PATH, additional_metadata=additional_metadata
        )

        # Print results
        print("✅ Document ingestion completed successfully!")
        print(f"📊 Document: {ingested_doc.original_filename}")
        print(f"📝 Total chunks: {len(ingested_doc.chunks)}")
        print(f"⏱️  Processing time: {ingested_doc.processing_time:.2f} seconds")
        print(f"💾 File size: {ingested_doc.file_size / 1024:.2f} KB")

        # Show collection info
        collection_info = engine.get_collection_info()
        print(
            f"🗂️  Tenant vectors in collection: {collection_info.get('tenant_vectors_count', 0)}"
        )

        # Show sample chunks (first 3)
        print("\n📋 Sample chunks:")
        for i, chunk in enumerate(ingested_doc.chunks[:3]):
            print(f"  Chunk {i + 1}: {chunk.content[:100]}...")

        return True

    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_ingestion():
    """Verify the document was ingested by searching for it."""

    TENANT_ID = "demo-tenant"

    try:
        print("\n🔍 Verifying ingestion by searching...")
        engine = get_ingestion_engine(tenant_id=TENANT_ID)

        # Search for content that might be in the PDF
        test_queries = [
            "introduction",
            "methodology",
            "conclusion",
            "abstract",
            "results",
        ]

        for query in test_queries:
            print(f"  Searching for: '{query}'")
            results = engine.search_within_tenant(query=query, k=2)
            if results:
                print(f"    ✅ Found {len(results)} results")
                for i, doc in enumerate(results):
                    print(f"      Result {i + 1}: {doc.page_content[:80]}...")
            else:
                print(f"    ⚠️  No results for '{query}'")
            print()

        # Get final count
        final_count = engine.get_tenant_document_count()
        print(f"📈 Final document count for tenant: {final_count}")

    except Exception as e:
        print(f"❌ Error during verification: {e}")


if __name__ == "__main__":
    print("🚀 PDF Vector Store Insertion Script")
    print("=" * 50)

    # Insert the PDF
    success = insert_pdf_document()

    if success:
        # Verify the insertion
        verify_ingestion()
        print("\n🎉 Script completed successfully!")
    else:
        print("\n💥 Script failed!")
        sys.exit(1)
