import json

from carag.core.config import get_settings

settings = get_settings()


def detect_hallucinations(
    query: str,
    llm_response: str,
    chunks: list[str],
    base_url: str = settings.HALLUCINATION_API_BASE,
) -> dict:
    """
    Simple function to detect hallucinations in LLM responses.

    Args:
        llm_response: The response from the LLM to check
        chunks: List of reference text chunks
        base_url: URL of the hallucination detection API

    Returns:
        API response as a dictionary
    """

    try:
        # # Simple request with timeout
        # response = httpx.post(
        #     endpoint,
        #     json=payload,
        #     timeout=60.0,  # Generous timeout
        # )
        # response.raise_for_status()
        # return response.json()
        return {"hallucination_result": "disabled"}

    except Exception as e:
        print(f"Error calling hallucination API: {e}")
        # Return a simple error structure
        return {"error": str(e), "status": "failed"}


# Even simpler version if you don't need error handling
def has_hallucinations(
    query: str,
    llm_response: str,
    chunks: list[str],
    api_url: str = settings.HALLUCINATION_API_BASE,
) -> bool:
    """
    A simple helper to check if any hallucinations were detected.

    Returns:
        True if hallucinated spans were found, False otherwise.
    """
    result = detect_hallucinations(query, llm_response, chunks, api_url)
    # The API returns a list of spans; if the list is not empty, hallucinations were found.
    if "hallucinated_spans" in result and result["hallucinated_spans"]:
        return True
    return False


if __name__ == "__main__":
    # Your API URL from the settings
    API_URL = settings.HALLUCINATION_API_BASE

    # --- Test Case 1: No Hallucination ---
    print("--- Running Test Case 1: No Hallucination ---")
    query_1 = "What is Python?"
    llm_response_1 = "Python is a high-level programming language created in 1991."
    chunks_1 = [
        "Python is a high-level, general-purpose programming language.",
        "It was created by Guido van Rossum and first released in 1991.",
    ]

    # This function now returns the full dictionary from your API
    result_1 = detect_hallucinations(query_1, llm_response_1, chunks_1, API_URL)
    print("Full API Response:")
    print(json.dumps(result_1, indent=2))

    # --- Test Case 2: With Hallucination ---
    print("\n--- Running Test Case 2: With Hallucination ---")
    query_2 = "What is the capital and population of France?"
    llm_response_2 = "The capital of France is Paris and the population is 69 million."
    chunks_2 = [
        "The capital of France is Paris.",
        "The population of France is 67 million.",
    ]

    result_2 = detect_hallucinations(query_2, llm_response_2, chunks_2, API_URL)
    print("Full API Response:")
    print(json.dumps(result_2, indent=2))

    # Example of using the boolean helper function
    is_hallucinated = has_hallucinations(query_2, llm_response_2, chunks_2, API_URL)
    print(f"\nWere hallucinations detected in Test Case 2? -> {is_hallucinated}")
