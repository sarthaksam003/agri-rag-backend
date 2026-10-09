import re


MAX_TTS_CHARS = 4500


def chunk_tts_text(text: str, max_chars: int = MAX_TTS_CHARS) -> list[str]:
    """
    Split prepared TTS text into chunks that stay below the Bhashini/API
    request limit while preserving sentence boundaries where possible.
    """

    if not text.strip():
        return []

    if len(text) <= max_chars:
        return [text.strip()]

    # Split into sentences while retaining the sentence-ending punctuation.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        # If adding this sentence still fits, keep it in the current chunk.
        candidate = (
            sentence
            if not current_chunk
            else f"{current_chunk} {sentence}"
        )

        if len(candidate) <= max_chars:
            current_chunk = candidate
            continue

        # Store the current chunk before starting another one.
        if current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""

        # A single sentence may itself be too long.
        # In that case, split it at whitespace boundaries.
        while len(sentence) > max_chars:
            split_at = sentence.rfind(" ", 0, max_chars)

            if split_at <= 0:
                split_at = max_chars

            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()

        current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks