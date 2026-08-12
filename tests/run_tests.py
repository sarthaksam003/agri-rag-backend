#!/usr/bin/env python3
"""
Test runner script for the Carag RAG API
"""

import os
import subprocess
import sys


def run_tests():
    """Run all tests with pytest"""
    # Add the src directory to Python path
    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    sys.path.insert(0, src_path)

    # Run pytest with specific configuration
    result = subprocess.run(
        [
            "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--cov=carag",
            "--cov-report=term-missing",
        ],
        cwd=os.path.dirname(__file__),
    )

    return result.returncode


if __name__ == "__main__":
    print("🚀 Running Carag RAG API Tests...")
    print("=" * 50)

    # Check if required services are running
    try:
        import requests

        health_check = requests.get("http://localhost:38779/health", timeout=5)
        if health_check.status_code == 200:
            print("✅ API server is running")
        else:
            print("❌ API server is not responding correctly")
            sys.exit(1)
    except:
        print("❌ API server is not running on port 38779")
        print("Please start the server with: python3 src/run_api.py")
        sys.exit(1)

    # Run the tests
    exit_code = run_tests()
    sys.exit(exit_code)
