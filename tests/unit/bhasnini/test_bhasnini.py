import asyncio

from carag.services.bhasini import synthesize, transcribe, translate


async def test_translate():
    """Test translation — easiest to verify, no audio needed."""
    print("--- Testing Translation (en → hi) ---")
    result = await translate(
        text="What is the capital of India?",
        source_language="en",
        target_language="hi",
    )
    print(f"Result: {result}")
    print()

    print("--- Testing Translation (hi → en) ---")
    result = await translate(
        text=result,
        source_language="hi",
        target_language="en",
    )
    print(f"Result: {result}")
    print()


async def test_tts():
    """Test TTS — generates audio bytes."""
    print("--- Testing TTS (hi) ---")
    audio_bytes = await synthesize(
        text="भारत की राजधानी नई दिल्ली है",
        target_language="hi",
    )
    print(f"Got {len(audio_bytes)} bytes of audio")

    # Write to file so you can play it
    with open("/tmp/bhasini_test.wav", "wb") as f:
        f.write(audio_bytes)
    print("Saved to /tmp/bhasini_test.wav")
    print()


async def test_asr():
    """Test ASR — needs the TTS output from above."""
    print("--- Testing ASR (hi) ---")
    with open("/tmp/bhasini_test.wav", "rb") as f:
        audio_bytes = f.read()

    text = await transcribe(
        audio_bytes=audio_bytes,
        source_language="hi",
    )
    print(f"Transcribed: {text}")
    print()


async def main():
    # Test in order: translate → tts → asr (asr uses tts output)
    await test_translate()
    await test_tts()
    await test_asr()


if __name__ == "__main__":
    asyncio.run(main())
