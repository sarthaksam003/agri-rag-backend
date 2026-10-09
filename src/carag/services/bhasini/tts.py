import base64
import html
import re

from loguru import logger

from carag.services.bhasini.client import call_inference
from carag.services.bhasini.schemas import TaskType
from carag.services.bhasini.audio import combine_wav_audio
from carag.services.bhasini.tts_chunker import chunk_tts_text


def prepare_tts_text(text: str, target_language: str = "en") -> str:
    """
    Convert the chatbot's Markdown/LaTeX response into speech-friendly text.

    This affects only the text sent to TTS. The original chatbot response
    shown in the UI remains unchanged.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # ---------------------------------------------------------
    # 1. Markdown tables
    # ---------------------------------------------------------
    lines = text.split("\n")
    output_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect a Markdown table header followed by a separator row.
        if (
            i + 1 < len(lines)
            and "|" in line
            and re.match(
                r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$",
                lines[i + 1],
            )
        ):
            headers = [
                cell.strip()
                for cell in line.strip().strip("|").split("|")
            ]

            i += 2

            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells = [
                    cell.strip()
                    for cell in lines[i].strip().strip("|").split("|")
                ]

                # Markdown responses may place source citations in an
                # additional trailing table cell, e.g. "| [2] |".
                # Citations are not needed for speech, so remove that cell
                # before comparing it with the table headers.
                if (
                    len(cells) == len(headers) + 1
                    and re.fullmatch(
                        r"(?:\[\d+\]\s*)+",
                        cells[-1],
                    )
                ):
                    cells.pop()

                # Ignore malformed rows rather than producing bad speech.
                if len(cells) == len(headers):
                    row_parts = []

                    for header, value in zip(headers, cells):
                        if header and value:
                            row_parts.append(f"{header}: {value}")
                        elif value:
                            row_parts.append(value)

                    if row_parts:
                        output_lines.append(". ".join(row_parts) + ".")

                i += 1

            continue

        output_lines.append(line)
        i += 1

    text = "\n".join(output_lines)

    # ---------------------------------------------------------
    # Remove source citations
    # ---------------------------------------------------------

    # [1], [2], [3][4], etc. are useful in the UI but should
    # not be spoken aloud.
    text = re.sub(
        r"\s*(?:\[\d+\]\s*)+",
        " ",
        text,
    )

    # Remove standalone citation bullets left behind after
    # removing the citation numbers.
    text = re.sub(
        r"(?m)^\s*[–—-]\s*$",
        "",
        text,
    )
    # ---------------------------------------------------------
    # Remove citation/table remnants
    # ---------------------------------------------------------
    text = re.sub(
        r"\s*\|\s*(?:\[\d+\](?:\s*\[\d+\])*)\s*\|?",
        " ",
        text,
    )

    # ---------------------------------------------------------
    # 2. Images and links
    # ---------------------------------------------------------
    # ![alt text](url) -> alt text
    text = re.sub(
        r"!\[([^\]]*)\]\([^)]+\)",
        r"\1",
        text,
    )

    # [link text](url) -> link text
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    # ---------------------------------------------------------
    # 3. Markdown headings
    # ---------------------------------------------------------
    text = re.sub(
        r"^\s{0,3}#{1,6}\s+",
        "",
        text,
        flags=re.MULTILINE,
    )

    # ---------------------------------------------------------
    # 4. Blockquotes
    # ---------------------------------------------------------
    text = re.sub(
        r"^\s*>\s?",
        "",
        text,
        flags=re.MULTILINE,
    )

    # ---------------------------------------------------------
    # 5. Bold / italic / strikethrough
    # ---------------------------------------------------------
    text = re.sub(r"\*\*\*(.*?)\*\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.*?)__", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"~~(.*?)~~", r"\1", text, flags=re.DOTALL)

    # Single * and _ used as emphasis.
    text = re.sub(r"(?<!\w)\*(.*?)\*(?!\w)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)_(.*?)_(?!\w)", r"\1", text, flags=re.DOTALL)

    # ---------------------------------------------------------
    # 6. Lists
    # ---------------------------------------------------------
    # Remove unordered-list markers.
    text = re.sub(
        r"^\s*[-+*]\s+",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Keep numbered-list numbering.
    text = re.sub(
        r"^\s*(\d+)\.\s+",
        r"\1. ",
        text,
        flags=re.MULTILINE,
    )

    # ---------------------------------------------------------
    # 7. Horizontal rules
    # ---------------------------------------------------------
    text = re.sub(
        r"^\s*(?:[-*_]\s*){3,}$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # ---------------------------------------------------------
    # 8. Code formatting
    # ---------------------------------------------------------
    # Inline code: `something` -> something
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Fenced code blocks: remove the fence but preserve the content.
    text = re.sub(r"```[^\n]*\n", "", text)
    text = text.replace("```", "")

    # ---------------------------------------------------------
    # 9. LaTeX delimiters
    # ---------------------------------------------------------
    # Convert the delimiters first; the actual mathematical notation
    # is handled below.
    text = text.replace(r"\[", "")
    text = text.replace(r"\]", "")
    text = text.replace(r"\(", "")
    text = text.replace(r"\)", "")

    # Remove display/inline dollar delimiters.
    text = text.replace("$$", "")
    text = text.replace("$", "")
    # ---------------------------------------------------------
    # 10a. Agricultural speech normalization
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Language-specific unit normalization
    # ---------------------------------------------------------

    if target_language == "hi":
        hindi_per_unit_replacements = [
            (r"मि\.ली\.?\s*/\s*हे\.?", "मिलीलीटर प्रति हेक्टेयर"),
            (r"मि\.ली\.?\s*/\s*किग्रा", "मिलीलीटर प्रति किलोग्राम"),
            (r"ली\.?\s*/\s*हे\.?", "लीटर प्रति हेक्टेयर"),
            (r"ली\.?\s*/\s*किग्रा", "लीटर प्रति किलोग्राम"),
            (r"किग्रा\s*/\s*हे\.?", "किलोग्राम प्रति हेक्टेयर"),
            (r"ग्रा\s*/\s*हे\.?", "ग्राम प्रति हेक्टेयर"),
            (r"मि\.ग्रा\.?\s*/\s*हे\.?", "मिलीग्राम प्रति हेक्टेयर"),
            (r"ग्रा\s*/\s*किग्रा", "ग्राम प्रति किलोग्राम"),
            (r"मि\.ग्रा\.?\s*/\s*किग्रा", "मिलीग्राम प्रति किलोग्राम"),
        ]

        for pattern, replacement in hindi_per_unit_replacements:
            text = re.sub(pattern, replacement, text)

        hindi_standalone_units = [
            (r"(?<=\d)\s*किग्रा(?!म)", " किलोग्राम"),
            (r"(?<=\d)\s*हे\.?", " हेक्टेयर"),
            (r"(?<=\d)\s*ग्रा(?!म)", " ग्राम"),
            (r"(?<=\d)\s*मि\.ग्रा\.?(?!म)", " मिलीग्राम"),
            (r"(?<=\d)\s*मि\.ली\.?(?!टर)", " मिलीलीटर"),
            (r"(?<=\d)\s*ली\.?(?!टर)", " लीटर"),
        ]

        for pattern, replacement in hindi_standalone_units:
            text = re.sub(pattern, replacement, text)
    # ---------------------------------------------------------
    # 1. Common scientific/agricultural abbreviations
    # ---------------------------------------------------------

    # Active ingredient abbreviation.
    text = re.sub(
        r"\ba\.i\.",
        "active ingredient",
        text,
        flags=re.IGNORECASE,
    )

    # Hectare abbreviation.
    text = re.sub(
        r"\bh\.a\.",
        "hectare",
        text,
        flags=re.IGNORECASE,
    )

    # ---------------------------------------------------------
    # 2. Compound agricultural units
    # ---------------------------------------------------------

    # Active ingredient + area/weight units.
    compound_units = [
        (
            r"\bg\s+active ingredient\s*/\s*m(?:²|\^2)\b",
            "grams of active ingredient per square metre",
        ),
        (
            r"\bg\s+active ingredient\s*/\s*ha\b",
            "grams of active ingredient per hectare",
        ),
        (
            r"\bkg\s+active ingredient\s*/\s*ha\b",
            "kilograms of active ingredient per hectare",
        ),
        (
            r"\bmg\s+active ingredient\s*/\s*kg\b",
            "milligrams of active ingredient per kilogram",
        ),
        (
            r"\bg\s+active ingredient\s*/\s*kg\b",
            "grams of active ingredient per kilogram",
        ),
    ]

    for pattern, replacement in compound_units:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    # ---------------------------------------------------------
    # 3. Per-unit expressions
    # ---------------------------------------------------------

    per_unit_replacements = [
        (r"\bmL\s*/\s*ha\b", "millilitres per hectare"),
        (r"\bml\s*/\s*ha\b", "millilitres per hectare"),
        (r"\bL\s*/\s*ha\b", "litres per hectare"),
        (r"\bl\s*/\s*ha\b", "litres per hectare"),
        (r"\bkg\s*/\s*ha\b", "kilograms per hectare"),
        (r"\bg\s*/\s*ha\b", "grams per hectare"),
        (r"\bmg\s*/\s*ha\b", "milligrams per hectare"),

        (r"\bmL\s*/\s*L\b", "millilitres per litre"),
        (r"\bml\s*/\s*l\b", "millilitres per litre"),
        (r"\bg\s*/\s*L\b", "grams per litre"),
        (r"\bg\s*/\s*l\b", "grams per litre"),
        (r"\bmg\s*/\s*L\b", "milligrams per litre"),
        (r"\bmg\s*/\s*l\b", "milligrams per litre"),

        (r"\bg\s*/\s*kg\b", "grams per kilogram"),
        (r"\bmg\s*/\s*kg\b", "milligrams per kilogram"),

        (r"\bmL\s*/\s*m(?:²|\^2)\b", "millilitres per square metre"),
        (r"\bml\s*/\s*m(?:²|\^2)\b", "millilitres per square metre"),
        (r"\bmg\s*/\s*m(?:²|\^2)\b", "milligrams per square metre"),
        (r"\bg\s*/\s*m(?:²|\^2)\b", "grams per square metre"),
        (r"\bmL\s+ha\b", "millilitres per hectare"),
        (r"\bml\s+ha\b", "millilitres per hectare"),
        (r"\bL\s+ha\b", "litres per hectare"),
        (r"\bl\s+ha\b", "litres per hectare"),
        (r"\bkg\s+ha\b", "kilograms per hectare"),
        (r"\bg\s+ha\b", "grams per hectare"),
        (r"\bmg\s+ha\b", "milligrams per hectare"),
    ]

    for pattern, replacement in per_unit_replacements:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    # ---------------------------------------------------------
    # 4. Negative-power notation
    # ---------------------------------------------------------

    # g m⁻² -> grams per square metre
    text = re.sub(
        r"\bg\s*m⁻²\b",
        "grams per square metre",
        text,
        flags=re.IGNORECASE,
    )

    # mg m⁻² -> milligrams per square metre
    text = re.sub(
        r"\bmg\s*m⁻²\b",
        "milligrams per square metre",
        text,
        flags=re.IGNORECASE,
    )

    # mL m⁻² -> millilitres per square metre
    text = re.sub(
        r"\bmL\s*m⁻²\b",
        "millilitres per square metre",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bm⁻²\b",
        "per square metre",
        text,
        flags=re.IGNORECASE,
    )
    # ---------------------------------------------------------
    # 5. Area and volume units
    # ---------------------------------------------------------

    text = re.sub(
        r"\bm²\b",
        "square metres",
        text,
    )

    text = re.sub(
        r"\bm³\b",
        "cubic metres",
        text,
    )

    # ---------------------------------------------------------
    # 3b. Common measurement per-unit expressions
    # ---------------------------------------------------------

    common_per_unit_replacements = [
        (r"(?:µ|μ)g\s*/\s*L\b", "micrograms per litre"),
        (r"mg\s*/\s*L\b", "milligrams per litre"),
        (r"g\s*/\s*L\b", "grams per litre"),
        (r"kg\s*/\s*L\b", "kilograms per litre"),

        (r"(?:µ|μ)g\s*/\s*kg\b", "micrograms per kilogram"),
        (r"mg\s*/\s*kg\b", "milligrams per kilogram"),
        (r"g\s*/\s*kg\b", "grams per kilogram"),
        (r"kg\s*/\s*kg\b", "kilograms per kilogram"),
    ]

    for pattern, replacement in common_per_unit_replacements:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )
    # ---------------------------------------------------------
    # 6. Unicode scientific notation
    # ---------------------------------------------------------

    # Convert chemical/scientific expressions containing Unicode
    # subscripts/superscripts into speech-friendly text.
    #
    # Examples:
    #   P₂O₅ -> P two O five
    #   K₂O  -> K two O
    #   N₂   -> N two
    #
    # This does not attempt to interpret the chemistry. It only
    # makes the notation readable to the TTS engine.

    subscript_digits = {
        "₀": "zero",
        "₁": "one",
        "₂": "two",
        "₃": "three",
        "₄": "four",
        "₅": "five",
        "₆": "six",
        "₇": "seven",
        "₈": "eight",
        "₉": "nine",
    }

    superscript_digits = {
        "⁰": "zero",
        "¹": "one",
        "²": "two",
        "³": "three",
        "⁴": "four",
        "⁵": "five",
        "⁶": "six",
        "⁷": "seven",
        "⁸": "eight",
        "⁹": "nine",
    }

    def replace_unicode_scientific_notation(match: re.Match) -> str:
        expression = match.group(0)

        result = []

        for character in expression:
            if character in subscript_digits:
                result.append(f" {subscript_digits[character]} ")
            elif character in superscript_digits:
                result.append(f" {superscript_digits[character]} ")
            else:
                result.append(character)

        return "".join(result)

    # Match a sequence containing letters/numbers with at least one
    # Unicode subscript or superscript digit.
    scientific_notation_pattern = (
        r"\b[A-Za-z0-9]+[₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]+[A-Za-z0-9]*\b"
    )

    text = re.sub(
        scientific_notation_pattern,
        replace_unicode_scientific_notation,
        text,
    )

    # ---------------------------------------------------------
    # 6. Unicode scientific notation
    # ---------------------------------------------------------

    subscript_digits = {
        "₀": "zero",
        "₁": "one",
        "₂": "two",
        "₃": "three",
        "₄": "four",
        "₅": "five",
        "₆": "six",
        "₇": "seven",
        "₈": "eight",
        "₉": "nine",
    }

    superscript_digits = {
        "⁰": "zero",
        "¹": "one",
        "²": "two",
        "³": "three",
        "⁴": "four",
        "⁵": "five",
        "⁶": "six",
        "⁷": "seven",
        "⁸": "eight",
        "⁹": "nine",
    }

    def replace_scientific_notation(match: re.Match) -> str:
        expression = match.group(0)
        spoken = []

        for char in expression:
            if char in subscript_digits:
                spoken.append(subscript_digits[char])
            elif char in superscript_digits:
                spoken.append(superscript_digits[char])
            else:
                spoken.append(char)

        return " ".join(spoken)

    scientific_notation_pattern = (
        r"\b[A-Za-z]+[₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]+[A-Za-z₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]*\b"
    )

    text = re.sub(
        scientific_notation_pattern,
        replace_scientific_notation,
        text,
    )

    if target_language == "hi":
        scientific_symbols = {
            "N": "एन",
            "P": "पी",
            "K": "के",
            "O": "ओ",
        }

        for symbol, pronunciation in scientific_symbols.items():
            text = re.sub(
                rf"(?<![A-Za-z]){symbol}(?=\s+(?:zero|one|two|three|four|five|six|seven|eight|nine)\b)",
                pronunciation,
                text,
            )

            text = re.sub(
                rf"(?<=\s){symbol}(?=\s|$)",
                pronunciation,
                text,
            )
    if target_language == "hi":
        scientific_numbers = {
            "zero": "ज़ीरो",
            "one": "वन",
            "two": "टू",
            "three": "थ्री",
            "four": "फोर",
            "five": "फाइव",
            "six": "सिक्स",
            "seven": "सेवन",
            "eight": "एट",
            "nine": "नाइन",
        }

        for number, pronunciation in scientific_numbers.items():
            text = re.sub(
                rf"\b{number}\b",
                pronunciation,
                text,
            )
    # ---------------------------------------------------------
    # 6. Standalone units
    # ---------------------------------------------------------

    # Only convert units when they directly follow a number.
    # This prevents words such as "e.g." from becoming "e.grams.".

    standalone_numeric_units = [
        # Length
        (r"(?<=\d)\s*(?:µ|μ)m\b", " micrometers"),
        (r"(?<=\d)\s*mm\b", " millimeters"),
        (r"(?<=\d)\s*cm\b", " centimeters"),
        (r"(?<=\d)\s*m\b", " meters"),
        (r"(?<=\d)\s*km\b", " kilometers"),

        # Mass
        (r"(?<=\d)\s*(?:µ|μ)g\b", " micrograms"),
        (r"(?<=\d)\s*mg\b", " milligrams"),
        (r"(?<=\d)\s*g\b", " grams"),
        (r"(?<=\d)\s*kg\b", " kilograms"),

        # Volume
        (r"(?<=\d)\s*(?:µ|μ)L\b", " microliters"),
        (r"(?<=\d)\s*mL\b", " milliliters"),
        (r"(?<=\d)\s*L\b", " liters"),

        # Existing agricultural unit
        (r"(?<=\d)\s*ha\b", " hectares"),
    ]

    for pattern, replacement in standalone_numeric_units:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    for pattern, replacement in standalone_numeric_units:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    # ---------------------------------------------------------
    # 7. Percentage
    # ---------------------------------------------------------

    text = text.replace("%", " percent ")

    # ---------------------------------------------------------
    # 8. Numeric ranges
    # ---------------------------------------------------------

    # 250–400 -> 250 to 400
    # 3–30 -> 3 to 30
    text = re.sub(
        r"(?<=\d)\s*[–—]\s*(?=\d)",
        " to ",
        text,
    )

    # ---------------------------------------------------------
    # 9. Common agricultural abbreviations
    # ---------------------------------------------------------

    agricultural_abbreviations = {
        "PPE": "personal protective equipment",
        "IPM": "integrated pest management",
        "PHI": "pre-harvest interval",
        "DAS": "days after sowing",
        "DAT": "days after treatment",
        "DAP": "days after planting",
    }

    for abbreviation, replacement in agricultural_abbreviations.items():
        text = re.sub(
            rf"\b{re.escape(abbreviation)}\b",
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(
        r"\b(personal protective equipment)\s*\(\s*personal protective equipment\s*\)",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    # ---------------------------------------------------------
    # 10. Basic mathematical symbols
    # ---------------------------------------------------------

    text = text.replace("×", " times ")
    text = text.replace("±", " plus or minus ")
    text = text.replace("≤", " less than or equal to ")
    text = text.replace("≥", " greater than or equal to ")

    # ---------------------------------------------------------
    # 11. Alphabetic hyphens
    # ---------------------------------------------------------

    # Treat hyphens between alphabetic words as spaces for speech.
    # Numeric ranges are handled separately above.
    text = re.sub(
        r"(?<=[A-Za-z])[--](?=[A-Za-z])",
        " ",
        text,
    )
    # ---------------------------------------------------------
    # 10. Common LaTeX expressions
    # ---------------------------------------------------------
    # \frac{a}{b} -> a divided by b
    text = re.sub(
        r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        r"\1 divided by \2",
        text,
    )

    # \sqrt{x} -> square root of x
    text = re.sub(
        r"\\sqrt\s*\{([^{}]*)\}",
        r"square root of \1",
        text,
    )

    # \text{something} -> something
    text = re.sub(
        r"\\text\s*\{([^{}]*)\}",
        r"\1",
        text,
    )

    # Superscripts: x^{2} -> x to the power of 2
    text = re.sub(
        r"\^\{([^{}]*)\}",
        r" to the power of \1",
        text,
    )

    # Simple superscripts: x^2 -> x to the power of 2
    text = re.sub(
        r"\^([A-Za-z0-9])",
        r" to the power of \1",
        text,
    )

    # Subscripts: x_{i} -> x subscript i
    text = re.sub(
        r"_\{([^{}]*)\}",
        r" subscript \1",
        text,
    )

    # Simple subscripts: x_i -> x subscript i
    text = re.sub(
        r"_([A-Za-z0-9])",
        r" subscript \1",
        text,
    )

    # ---------------------------------------------------------
    # 11. HTML
    # ---------------------------------------------------------
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)

    # ---------------------------------------------------------
    # 12. Collapse excessive whitespace
    # ---------------------------------------------------------
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces immediately before punctuation.
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Collapse 3+ newlines to 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Restore spaces accidentally removed around Markdown fragments.
    text = text.replace("beforecrop", "before crop")
    text = text.replace("Mosquitoes.Typical", "Mosquitoes. Typical")
    text = text.replace("Cockroaches.Typical", "Cockroaches. Typical")
    return text.strip()


async def synthesize(
    text: str,
    target_language: str,
    gender: str = "female",
) -> bytes:
    prepared_text = prepare_tts_text(text, target_language)
    text_chunks = chunk_tts_text(prepared_text)

    if not text_chunks:
        raise ValueError("No text available for TTS")

    logger.info(
        f"TTS prepared text: {len(prepared_text)} characters, "
        f"{len(text_chunks)} chunk(s)"
    )

    audio_chunks: list[bytes] = []

    for index, chunk in enumerate(text_chunks, start=1):
        logger.info(
            f"TTS chunk {index}/{len(text_chunks)}: "
            f"{len(chunk)} characters"
        )

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "tts",
                    "config": {
                        "language": {"sourceLanguage": target_language},
                        "gender": gender,
                    },
                }
            ],
            "inputData": {"input": [{"source": chunk}]},
        }

        result = await call_inference(
            task_type=TaskType.TTS,
            source_language=target_language,
            target_language=None,
            payload=payload,
        )

        audio_b64 = result["pipelineResponse"][0]["audio"][0]["audioContent"]
        audio_bytes = base64.b64decode(audio_b64)

        audio_chunks.append(audio_bytes)

        logger.info(
            f"TTS chunk {index}/{len(text_chunks)} result: "
            f"{len(audio_bytes)} bytes"
        )

    combined_audio = combine_wav_audio(audio_chunks)

    logger.info(
        f"TTS combined result ({target_language}): "
        f"{len(combined_audio)} bytes from {len(audio_chunks)} chunk(s)"
    )

    return combined_audio
