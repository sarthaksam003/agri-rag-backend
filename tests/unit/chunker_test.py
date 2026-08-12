from carag.services.document_ingestion.chunker import get_chunker

with open("fixed_tables.md", "r") as f:
    markdown = f.read()

chunker = get_chunker()
chunks = chunker.chunk(markdown)


with open("chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟 CHUNK NUM {i} 🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟\n")
        f.write(chunk)
        f.write("\n🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑 CHUNK END 🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑\n\n")
