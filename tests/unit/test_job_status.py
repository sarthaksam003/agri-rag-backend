"""
Test script for job status endpoint - Bug 1 verification
"""

import uuid

import requests

# Configuration
BASE_URL = "http://localhost:38779"
TENANT_ID = "demo-tenant"


def test_nonexistent_job():
    """Test that non-existent job returns proper error"""
    print("🧪 Testing non-existent job status...")

    # Generate a random UUID that definitely doesn't exist
    fake_job_id = str(uuid.uuid4())

    response = requests.get(
        f"{BASE_URL}/api/v1/documents/jobs/{fake_job_id}",
        headers={"x-tenant-id": TENANT_ID},
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    # The fixed version should return status indicating not found
    data = response.json()
    if data.get("status") == "not_found":
        print("✅ PASS: Non-existent job correctly returns 'not_found'")
    else:
        print("❌ FAIL: Non-existent job should return 'not_found' status")


def test_real_job_flow():
    """Test complete job flow with actual file upload"""
    print("\n🧪 Testing real job flow...")

    # First, upload a test file to get a real job ID
    test_file_path = "data/1.pdf"

    try:
        with open(test_file_path, "rb") as f:
            files = {"files": ("test.pdf", f, "application/pdf")}
            response = requests.post(
                f"{BASE_URL}/api/v1/documents/ingest",
                headers={"x-tenant-id": TENANT_ID},
                files=files,
            )

        if response.status_code == 202:
            data = response.json()
            job_id = data["processed_files"][0]["job_id"]
            print(f"✅ File uploaded successfully. Job ID: {job_id}")

            # Now test the job status endpoint
            print(f"🔍 Checking status for real job: {job_id}")
            status_response = requests.get(
                f"{BASE_URL}/api/v1/documents/jobs/{job_id}",
                headers={"x-tenant-id": TENANT_ID},
            )

            status_data = status_response.json()
            print(f"Job Status: {status_data}")

            if status_data.get("status") in ["processing", "completed", "failed"]:
                print("✅ PASS: Real job returns valid status")
            else:
                print("❌ FAIL: Real job should return valid status")

        else:
            print(f"❌ File upload failed: {response.status_code} - {response.text}")

    except FileNotFoundError:
        print("❌ Test file not found. Please ensure data/1.pdf exists")
    except Exception as e:
        print(f"❌ Error during real job test: {e}")


if __name__ == "__main__":
    print("🚀 Starting Job Status Tests")
    print("=" * 50)

    test_nonexistent_job()
    test_real_job_flow()
