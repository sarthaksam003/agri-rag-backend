from carag.core.constants import get_constants

constants = get_constants()
llm = constants.llm

response = llm.invoke(
    "Generate three alternative search queries for: What is a transformer?"
)

print("CONTENT:")
print(response.content)

print("\nADDITIONAL KWARGS:")
print(response.additional_kwargs)

print("\nRESPONSE METADATA:")
print(response.response_metadata)