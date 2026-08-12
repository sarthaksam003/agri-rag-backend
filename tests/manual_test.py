"""
Manual test script for the Carag RAG API
This avoids pytest import issues and tests endpoints directly
"""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Add src to path and load environment
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

BASE_URL = "http://localhost:38779"
TENANT_HEADERS = {"X-Tenant-ID": "demo-tenant"}


def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    assert response.status_code == 200, f"Health check failed: {response.status_code}"
    assert response.json()["status"] == "healthy"
    print("✅ Health check passed")


def test_list_documents():
    """Test listing documents"""
    print("🔍 Testing document listing...")
    response = requests.get(
        f"{BASE_URL}/api/v1/documents/", headers=TENANT_HEADERS, timeout=10
    )
    assert response.status_code == 200, f"List documents failed: {response.status_code}"
    data = response.json()
    print(f"✅ Document listing: {data}")


def test_ingest_pdf():
    """Test PDF ingestion"""
    pdf_path = Path("data/1.pdf")
    if not pdf_path.exists():
        print("❌ Sample PDF not found, skipping ingestion test")
        return False

    print(f"🔍 Testing PDF ingestion with {pdf_path}...")

    with open(pdf_path, "rb") as f:
        files = {"files": (pdf_path.name, f, "application/pdf")}
        response = requests.post(
            f"{BASE_URL}/api/v1/documents/ingest",
            headers=TENANT_HEADERS,
            files=files,
            timeout=30,
        )

    assert response.status_code == 200, f"Ingestion failed: {response.status_code}"
    data = response.json()
    print(f"✅ PDF ingestion: {data}")
    return True


def test_chat():
    """Test chat endpoint"""
    print("🔍 Testing chat endpoint...")

    chat_data = {
        "message": "Hello, can you tell me about the documents?",
        "session_id": "manual-test-session",
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/chat/message",
        headers=TENANT_HEADERS,
        json=chat_data,
        timeout=30,
    )

    assert response.status_code == 200, f"Chat failed: {response.status_code}"
    data = response.json()
    print(f"✅ Chat response received (length: {len(data['response'])})")
    print(f"   Session: {data['session_id']}")
    print(f"   Documents retrieved: {data['retrieved_documents_count']}")


def test_delete_all():
    """Test deleting all tenant data"""
    print("🔍 Testing delete all endpoint...")

    response = requests.delete(
        f"{BASE_URL}/api/v1/documents/tenant/all", headers=TENANT_HEADERS, timeout=10
    )

    assert response.status_code == 200, f"Delete all failed: {response.status_code}"
    data = response.json()
    print(f"✅ Delete all: {data}")


def main():
    """Run all manual tests"""
    print("🚀 Starting Manual API Tests")
    print("=" * 50)

    try:
        # Verify environment
        required_vars = [
            "EMBEDDINGS_API_MODEL",
            "OPENAI_API_MODEL",
            "QDRANT_HOST",
            "QDRANT_PORT",
        ]
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                print(f"❌ Missing environment variable: {var}")
                return
            print(f"✅ {var}: {value}")

        # Run tests
        test_health()
        test_list_documents()

        if test_ingest_pdf():
            test_chat()

        test_delete_all()

        print("🎉 All manual tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
