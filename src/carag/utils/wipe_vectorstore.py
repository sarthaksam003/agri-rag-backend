"""Quick script to wipe Qdrant - use with extreme caution!"""

from carag.core.constants import get_constants

constants = get_constants()

# List and delete all collections
collections = constants.vectorstore.get_collections().collections
print(f"Found {len(collections)} collections")

for coll in collections:
    print(f"Deleting: {coll.name}")
    constants.vectorstore.delete_collection(collection_name=coll.name)

print("Done! Vectorstore wiped.")
