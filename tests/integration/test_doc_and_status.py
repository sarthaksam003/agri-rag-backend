"""
Comprehensive integration test for both bugs
"""

import os
import time
import uuid

import requests

BASE_URL = "http://localhost:38779"
TENANT_ID = "demo-tenant"


def test_both_bugs_together():
    """Test both bugs in an integrated scenario"""
    print("🔗 Testing Both Bugs Together")
    print("=" * 60)

    # Test 1: Verify non-existent job handling
    print("\n1. Testing non-existent job handling...")
    fake_job_id = str(uuid.uuid4())
    response = requests.get(
        f"{BASE_URL}/api/v1/documents/jobs/{fake_job_id}",
        headers={"x-tenant-id": TENANT_ID},
    )

    job_data = response.json()
    if job_data.get("status") == "not_found":
        print("✅ Bug 1 FIXED: Non-existent jobs return 'not_found'")
    else:
        print("❌ Bug 1 NOT FIXED: Non-existent jobs should return 'not_found'")

    # Test 2: Upload and delete document
    print("\n2. Testing document upload and deletion...")

    # Upload document
    test_file = "data/1.pdf"
    if not os.path.exists(test_file):
        print("❌ Test file not found. Using data/2.pdf instead...")
        test_file = "data/2.pdf"
        if not os.path.exists(test_file):
            print("❌ No test files found. Please add PDF files to data/ directory")
            return

    with open(test_file, "rb") as f:
        files = {"files": ("integration_test.pdf", f, "application/pdf")}
        upload_response = requests.post(
            f"{BASE_URL}/api/v1/documents/ingest",
            headers={"x-tenant-id": TENANT_ID},
            files=files,
        )

    if upload_response.status_code != 202:
        print("❌ Upload failed. Cannot proceed with deletion test.")
        return

    upload_data = upload_response.json()
    document_id = upload_data["processed_files"][0]["document_id"]
    job_id = upload_data["processed_files"][0]["job_id"]

    print(f"📄 Uploaded document: {document_id}")

    # Wait for ingestion
    print("⏳ Waiting for ingestion...")
    time.sleep(10)  # Longer wait for reliability

    # Delete the document
    delete_response = requests.delete(
        f"{BASE_URL}/api/v1/documents/{document_id}", headers={"x-tenant-id": TENANT_ID}
    )

    if delete_response.status_code == 200:
        delete_data = delete_response.json()
        deleted_chunks = delete_data.get("deleted_chunks", 0)

        if deleted_chunks > 0:
            print("✅ Bug 2 FIXED: Document deletion reports correct chunk count")
        else:
            print("❌ Bug 2 NOT FIXED: Document deletion shows 0 chunks deleted")

        # Verify document is gone
        list_response = requests.get(
            f"{BASE_URL}/api/v1/documents/", headers={"x-tenant-id": TENANT_ID}
        )

        list_data = list_response.json()
        documents = list_data.get("data", {}).get("documents", [])
        document_ids = [doc["id"] for doc in documents]

        if document_id not in document_ids:
            print("✅ Document properly removed from listing")
        else:
            print("❌ Document still appears in listing after deletion")

    elif delete_response.status_code == 404:
        print("❌ Document not found for deletion (might be ingestion issue)")
    else:
        print(f"❌ Delete failed: {delete_response.status_code}")


def quick_health_check():
    """Quick health check of the API"""
    print("🏥 Quick Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy")
            return True
        else:
            print("❌ API health check failed")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Comprehensive Integration Tests")
    print("=" * 60)

    if not quick_health_check():
        print("❌ Cannot proceed - API is not accessible")
        exit(1)

    test_both_bugs_together()

    print("\n" + "=" * 60)
    print("📊 Test Summary Complete")
