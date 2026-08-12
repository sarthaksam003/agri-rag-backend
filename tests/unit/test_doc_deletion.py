"""
Test script for document deletion - Bug 2 verification
"""

import os
import time

import requests

# Configuration
BASE_URL = "http://localhost:38779"
TENANT_ID = "demo-tenant"


def upload_test_document():
    """Upload a test document and return its document_id"""
    print("📤 Uploading test document...")

    test_file_path = "data/1.pdf"

    if not os.path.exists(test_file_path):
        print("❌ Test file data/1.pdf not found")
        return None

    try:
        with open(test_file_path, "rb") as f:
            files = {"files": ("test_document.pdf", f, "application/pdf")}
            response = requests.post(
                f"{BASE_URL}/api/v1/documents/ingest",
                headers={"x-tenant-id": TENANT_ID},
                files=files,
            )

        if response.status_code == 202:
            data = response.json()
            document_id = data["processed_files"][0]["document_id"]
            job_id = data["processed_files"][0]["job_id"]
            print(f"✅ Document uploaded. Document ID: {document_id}, Job ID: {job_id}")

            # Wait for ingestion to complete
            print("⏳ Waiting for ingestion to complete...")
            return wait_for_ingestion_completion(job_id, document_id)
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None


def wait_for_ingestion_completion(job_id, document_id, max_wait=60):
    """Wait for the ingestion job to complete"""
    start_time = time.time()

    while time.time() - start_time < max_wait:
        response = requests.get(
            f"{BASE_URL}/api/v1/documents/jobs/{job_id}",
            headers={"x-tenant-id": TENANT_ID},
        )

        if response.status_code == 200:
            data = response.json()
            status = data.get("status")

            if status == "completed":
                print("✅ Ingestion completed successfully")
                return document_id
            elif status == "failed":
                print(
                    f"❌ Ingestion failed: {data.get('error_message', 'Unknown error')}"
                )
                return None
            elif status == "processing":
                print("⏳ Still processing...")
                time.sleep(2)
            else:
                print(f"❓ Unexpected status: {status}")
                time.sleep(2)
        else:
            print(f"❌ Job status check failed: {response.status_code}")
            time.sleep(2)

    print("❌ Ingestion timeout")
    return None


def list_documents():
    """List all documents for the tenant"""
    print("📋 Listing documents...")

    response = requests.get(
        f"{BASE_URL}/api/v1/documents/", headers={"x-tenant-id": TENANT_ID}
    )

    if response.status_code == 200:
        data = response.json()
        documents = data.get("data", {}).get("documents", [])
        total = data.get("data", {}).get("total_documents", 0)

        print(f"📊 Found {total} documents:")
        for doc in documents:
            print(f"  - ID: {doc['id']}, Filename: {doc['filename']}")

        return documents, total
    else:
        print(f"❌ List documents failed: {response.status_code} - {response.text}")
        return [], 0


def delete_document(document_id):
    """Delete a specific document"""
    print(f"🗑️  Deleting document: {document_id}")

    response = requests.delete(
        f"{BASE_URL}/api/v1/documents/{document_id}", headers={"x-tenant-id": TENANT_ID}
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Delete response: {data}")
        return data
    elif response.status_code == 404:
        print(f"❌ Document not found: {response.json()}")
        return response.json()
    else:
        print(f"❌ Delete failed: {response.status_code} - {response.text}")
        return None


def test_nonexistent_document_deletion():
    """Test deleting a document that doesn't exist"""
    print("\n🧪 Testing deletion of non-existent document...")

    fake_document_id = "this-does-not-exist-12345"

    response_data = delete_document(fake_document_id)

    if (
        response_data
        and "error" in str(response_data).lower()
        or response_data.get("deleted_chunks") == 0
    ):
        print("✅ PASS: Non-existent document deletion handled correctly")
    else:
        print("❌ FAIL: Should return error for non-existent document")


def test_document_lifecycle():
    """Test complete document lifecycle"""
    print("🚀 Testing Document Lifecycle")
    print("=" * 50)

    # Step 1: List initial documents
    initial_docs, initial_count = list_documents()

    # Step 2: Upload a test document
    document_id = upload_test_document()
    if not document_id:
        print("❌ Cannot proceed - document upload failed")
        return

    # Wait a moment for the document to be fully processed
    time.sleep(3)

    # Step 3: Verify document appears in list
    docs_after_upload, count_after_upload = list_documents()

    if count_after_upload > initial_count:
        print("✅ PASS: Document count increased after upload")
    else:
        print("❌ FAIL: Document count should increase after upload")

    # Step 4: Delete the document
    delete_response = delete_document(document_id)

    if delete_response and delete_response.get("deleted_chunks", 0) > 0:
        print("✅ PASS: Document deletion reported chunks deleted")
    else:
        print("❌ FAIL: No chunks reported as deleted")

    # Wait a moment for deletion to propagate
    time.sleep(2)

    # Step 5: Verify document is removed from list
    final_docs, final_count = list_documents()

    if final_count == initial_count:
        print("✅ PASS: Document count returned to initial after deletion")
    else:
        print(f"❌ FAIL: Expected {initial_count} documents, got {final_count}")

    # Step 6: Try to delete the same document again (should fail)
    print("\n🧪 Testing re-deletion of same document...")
    second_delete_response = delete_document(document_id)

    if second_delete_response and (
        "not found" in str(second_delete_response).lower()
        or second_delete_response.get("deleted_chunks") == 0
    ):
        print("✅ PASS: Re-deletion correctly handled")
    else:
        print("❌ FAIL: Re-deletion should return error")


if __name__ == "__main__":
    print("🚀 Starting Document Deletion Tests")
    print("=" * 50)

    test_document_lifecycle()
    test_nonexistent_document_deletion()
