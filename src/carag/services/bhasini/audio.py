import io
import struct


def _parse_wav(wav_bytes: bytes) -> tuple[bytes, bytes]:
    """
    Extract the WAV format chunk and audio data chunk.

    This works with PCM and IEEE floating-point WAV files without
    requiring the Python wave module to decode the audio format.
    """
    if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("Invalid WAV file")

    position = 12
    fmt_chunk = None
    data_chunk = None

    while position + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[position:position + 4]
        chunk_size = struct.unpack_from(
            "<I",
            wav_bytes,
            position + 4,
        )[0]

        chunk_start = position + 8
        chunk_end = chunk_start + chunk_size

        if chunk_end > len(wav_bytes):
            raise ValueError("Invalid WAV chunk size")

        chunk_data = wav_bytes[chunk_start:chunk_end]

        if chunk_id == b"fmt ":
            fmt_chunk = chunk_data
        elif chunk_id == b"data":
            data_chunk = chunk_data

        # RIFF chunks are padded to an even number of bytes.
        position = chunk_end + (chunk_size % 2)

    if fmt_chunk is None:
        raise ValueError("WAV file has no fmt chunk")

    if data_chunk is None:
        raise ValueError("WAV file has no data chunk")

    return fmt_chunk, data_chunk


def combine_wav_audio(audio_chunks: list[bytes]) -> bytes:
    """
    Concatenate WAV audio chunks into a single WAV file.

    All chunks must have the same WAV format.
    """

    if not audio_chunks:
        raise ValueError("No audio chunks provided")

    if len(audio_chunks) == 1:
        return audio_chunks[0]

    first_fmt, first_data = _parse_wav(audio_chunks[0])

    all_audio_data = [first_data]

    for audio_bytes in audio_chunks[1:]:
        fmt_chunk, data_chunk = _parse_wav(audio_bytes)

        if fmt_chunk != first_fmt:
            raise ValueError(
                "TTS audio chunks have incompatible WAV formats"
            )

        all_audio_data.append(data_chunk)

    combined_data = b"".join(all_audio_data)

    # Preserve the original WAV format and construct a new
    # RIFF/WAVE container around the combined audio data.
    fmt_chunk_header = struct.pack(
        "<4sI",
        b"fmt ",
        len(first_fmt),
    )

    data_chunk_header = struct.pack(
        "<4sI",
        b"data",
        len(combined_data),
    )

    wav_body = (
        fmt_chunk_header
        + first_fmt
        + data_chunk_header
        + combined_data
    )

    # RIFF size excludes the initial 8 bytes ("RIFF" + size).
    riff_size = 4 + len(wav_body)

    output = io.BytesIO()
    output.write(b"RIFF")
    output.write(struct.pack("<I", riff_size))
    output.write(b"WAVE")
    output.write(wav_body)

    return output.getvalue()