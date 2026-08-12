"""
Test-specific configuration to ensure environment variables are properly loaded
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the root .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


# Verify critical environment variables
def verify_environment():
    """Verify that all required environment variables are set"""
    required_vars = {
        "EMBEDDINGS_API_MODEL": "Embeddings model name",
        "OPENAI_API_MODEL": "OpenAI model name",
        "QDRANT_HOST": "Qdrant host",
        "QDRANT_PORT": "Qdrant port",
        "OPENAI_API_BASE": "OpenAI API base URL",
        "EMBEDDINGS_API_BASE": "Embeddings API base URL",
    }

    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")

    if missing_vars:
        raise EnvironmentError(
            "Missing required environment variables:\n"
            + "\n".join(f"  - {var}" for var in missing_vars)
            + f"\n\nPlease check your .env file at: {env_path}"
        )

    print("✅ All required environment variables are set")


if __name__ == "__main__":
    verify_environment()
