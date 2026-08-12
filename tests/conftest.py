import os
import sys
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

# Load environment variables from the root .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Now import the app after environment is loaded
from carag.api.main import app


@pytest.fixture
def client():
    """Test client for FastAPI app"""
    return TestClient(app)


@pytest.fixture
def sample_pdf_1():
    """Path to sample PDF 1"""
    pdf_path = Path("data/1.pdf")
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF 1 not found at {pdf_path}")
    return pdf_path


@pytest.fixture
def sample_pdf_2():
    """Path to sample PDF 2"""
    pdf_path = Path("data/2.pdf")
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF 2 not found at {pdf_path}")
    return pdf_path


@pytest.fixture
def tenant_headers():
    """Default tenant headers"""
    return {"X-Tenant-ID": "demo-tenant"}


@pytest.fixture
def temp_pdf_files(sample_pdf_1, sample_pdf_2):
    """Create temporary copies of PDF files for testing"""
    temp_files = []
    for pdf_path in [sample_pdf_1, sample_pdf_2]:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_path.read_bytes())
            temp_files.append(temp_file.name)

    yield temp_files

    # Cleanup
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests"""
    # Verify that required environment variables are set
    required_vars = [
        "EMBEDDINGS_API_MODEL",
        "OPENAI_API_MODEL",
        "QDRANT_HOST",
        "QDRANT_PORT",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        pytest.fail(f"Missing required environment variables: {missing_vars}")

    print("✅ Test environment setup complete")
