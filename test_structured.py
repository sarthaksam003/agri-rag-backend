from carag.core.constants import get_constants
from pydantic import BaseModel, Field


class QueryOutput(BaseModel):
    questions: list[str] = Field(
        description="List of alternative search queries"
    )


constants = get_constants()
llm = constants.llm

messages = [
    {
        "role": "user",
        "content": (
            "Generate three alternative search queries for: "
            "What is a transformer?"
        ),
    }
]


for method in ["json_schema", "function_calling"]:
    print("\n" + "=" * 60)
    print(f"TESTING METHOD: {method}")
    print("=" * 60)

    try:
        structured_llm = llm.with_structured_output(
            QueryOutput,
            method=method,
        )

        response = structured_llm.invoke(messages)

        print("SUCCESS")
        print("Response:", response)

    except Exception as e:
        print("FAILED")
        print(type(e).__name__)
        print(e)
