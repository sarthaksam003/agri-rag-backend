from carag.services.bhasini.tts_chunker import chunk_tts_text


text = """
This is the first sentence. This is the second sentence. This is the third sentence.
"""


chunks = chunk_tts_text(text, max_chars=50)

print("Number of chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print()
    print(f"----- CHUNK {index} -----")
    print("Characters:", len(chunk))
    print(chunk)