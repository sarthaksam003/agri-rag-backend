"""Convert MinerU HTML tables into Markdown tables for table-aware chunking.

This module deliberately handles representation only. It does not merge tables
across PDF pages or infer row relationships between separate table fragments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from bs4 import BeautifulSoup, Tag


_TABLE_TAG_RE = re.compile(r"</?table\b[^>]*>", re.IGNORECASE)
_HEADER_HINTS = (
    "institution",
    "variety",
    "special feature",
    "feature",
    "characteristic",
    "description",
    "parameter",
    "treatment",
    "dose",
    "year",
    "name",
    "category",
    "item",
    "crop",
    "stage",
    "pest",
    "disease",
    "observation",
    "measurement",
    "particular",
    "component",
)


@dataclass(frozen=True)
class TableSourceSpan:
    """Map a converted/reconstructed table to source offsets and PDF pages."""

    source_start: int
    source_end: int
    markdown_start: int
    markdown_end: int
    source_pages: tuple[int, ...] = ()


@dataclass(frozen=True)
class HTMLTableConversionResult:
    """Result, counters and source mappings from table conversion."""

    markdown: str
    tables_found: int
    tables_converted: int
    tables_skipped: int
    replacements: tuple[TableSourceSpan, ...] = ()


SOURCE_RANGE_MARKER_RE = re.compile(
    r"<!--\s*AGRICHAT_SOURCE_RANGE:(\d+):(\d+)"
    r"(?::PAGES=([0-9,]+))?\s*-->"
)




def map_normalized_offset_to_source(
    result: HTMLTableConversionResult, offset: int
) -> int:
    """Map an offset in normalized Markdown back to the original source text.

    Offsets inside a converted HTML table map to that table's original start;
    offsets after converted tables account for the length differences introduced
    by HTML-to-Markdown conversion.
    """
    if offset < 0:
        raise ValueError("offset must be non-negative")

    cumulative_shift = 0
    for replacement in result.replacements:
        if offset < replacement.markdown_start:
            break
        if offset < replacement.markdown_end:
            return replacement.source_start
        if offset == replacement.markdown_end:
            return replacement.source_end

        original_length = replacement.source_end - replacement.source_start
        normalized_length = replacement.markdown_end - replacement.markdown_start
        cumulative_shift += normalized_length - original_length

    return max(0, offset - cumulative_shift)


def _iter_table_spans(text: str) -> Iterable[tuple[int, int, str]]:
    """Yield balanced table tag spans, including nested table tags."""
    depth = 0
    start: int | None = None

    for match in _TABLE_TAG_RE.finditer(text):
        tag = match.group(0)
        is_closing = tag.lstrip().startswith("</")
        is_self_closing = tag.rstrip().endswith("/>")

        if is_closing:
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                yield start, match.end(), text[start : match.end()]
                start = None
        elif not is_self_closing:
            if depth == 0:
                start = match.start()
            depth += 1


def _cell_text(cell: Tag) -> str:
    """Extract readable cell text while retaining line-break separation."""
    for br in cell.find_all("br"):
        br.replace_with(" ")
    value = cell.get_text(" ", strip=True)
    value = re.sub(r"\s+", " ", value).strip()
    # Escape pipes so cell values cannot accidentally create extra columns.
    return value.replace("|", r"\|")


def _positive_span(cell: Tag, attribute: str) -> int:
    try:
        value = int(cell.get(attribute, 1))
    except (TypeError, ValueError):
        return 1
    return max(1, value)


def _table_rows(table: Tag) -> list[list[str]]:
    """Expand table rows into a rectangular cell matrix.

    Rowspan text is repeated in the affected rows because HTML defines that cell
    as applying to those rows. Colspan text is placed in the first covered column
    and the remaining covered columns are left empty; no content is invented.
    """
    html_rows = [
        row
        for row in table.find_all("tr")
        if row.find_parent("table") is table
    ]
    if not html_rows:
        return []

    active_rowspans: dict[int, tuple[str, int]] = {}
    expanded_rows: list[list[str]] = []

    for row in html_rows:
        slots: dict[int, str] = {}
        next_rowspans: dict[int, tuple[str, int]] = {}

        for column, (value, remaining_rows) in active_rowspans.items():
            slots[column] = value
            if remaining_rows > 1:
                next_rowspans[column] = (value, remaining_rows - 1)

        cells = [
            cell
            for cell in row.find_all(["th", "td"])
            if cell.find_parent("tr") is row and cell.find_parent("table") is table
        ]

        column = 0
        for cell in cells:
            while column in slots:
                column += 1

            colspan = _positive_span(cell, "colspan")
            rowspan = _positive_span(cell, "rowspan")
            value = _cell_text(cell)

            for offset in range(colspan):
                target_column = column + offset
                target_value = value if offset == 0 else ""
                slots[target_column] = target_value

                if rowspan > 1:
                    next_rowspans[target_column] = (target_value, rowspan - 1)

            column += colspan

        if slots:
            width = max(slots) + 1
            expanded_rows.append([slots.get(index, "") for index in range(width)])

        active_rowspans = next_rowspans

    if not expanded_rows:
        return []

    width = max(len(row) for row in expanded_rows)
    return [row + [""] * (width - len(row)) for row in expanded_rows]


def _looks_like_header(first_row: list[str], first_html_row: Tag) -> bool:
    """Recognize explicit TH rows or obvious all-TD MinerU header rows."""
    if first_html_row.find("th", recursive=False) is not None:
        return True

    normalized_cells = [re.sub(r"[^a-z0-9]+", " ", value.lower()).strip() for value in first_row]
    non_empty = [value for value in normalized_cells if value]
    matched_columns = sum(
        any(hint in value for hint in _HEADER_HINTS)
        for value in non_empty
    )
    return len(non_empty) >= 2 and matched_columns >= 2


def _markdown_cell(value: str) -> str:
    # Markdown tables cannot safely contain literal newlines within a cell.
    return re.sub(r"\s+", " ", value).strip()


def html_table_to_markdown(table_html: str) -> str | None:
    """Convert one HTML table to a pipe table; return None if it has no rows."""
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if table is None:
        return None

    rows = _table_rows(table)
    if not rows:
        return None

    first_html_row = next(
        row
        for row in table.find_all("tr")
        if row.find_parent("table") is table
    )
    has_header = _looks_like_header(rows[0], first_html_row)

    width = max(len(row) for row in rows)
    if has_header:
        header = rows[0]
        data_rows = rows[1:]
    else:
        # Continuation fragments often have no repeated source header. Generic
        # positional labels keep every source row as data until Stage B rebuilds
        # the logical cross-page table using the original document context.
        header = [f"Column {index + 1}" for index in range(width)]
        data_rows = rows

    header = header + [""] * (width - len(header))
    lines = [
        "| " + " | ".join(_markdown_cell(value) for value in header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]

    for row in data_rows:
        padded = row + [""] * (width - len(row))
        lines.append("| " + " | ".join(_markdown_cell(value) for value in padded) + " |")

    caption = table.find("caption", recursive=False)
    caption_text = _cell_text(caption) if caption is not None else ""
    if caption_text:
        lines.insert(0, f"**{caption_text}**")

    return "\n".join(lines)




@dataclass(frozen=True)
class _ConvertedTable:
    source_start: int
    source_end: int
    markdown: str
    page_number: int | None
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    has_real_header: bool
    caption: str = ""


def _page_for_offset(
    offset: int, page_boundaries: list[dict[str, int]] | None
) -> int | None:
    if not page_boundaries:
        return None
    for boundary in page_boundaries:
        if boundary["start"] <= offset < boundary["end"]:
            return int(boundary["page_number"])
    return None


def _source_pages_for_range(
    start: int,
    end: int,
    page_boundaries: list[dict[str, int]] | None,
) -> tuple[int, ...]:
    if not page_boundaries:
        return ()
    pages = [
        int(boundary["page_number"])
        for boundary in page_boundaries
        if boundary["start"] < end and boundary["end"] > start
    ]
    return tuple(dict.fromkeys(pages))


def _split_markdown_row(line: str) -> list[str]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    cells = re.split(r"(?<!\\)\|", value)
    return [cell.replace(r"\|", "|").strip() for cell in cells]


def _parse_converted_table(markdown: str) -> tuple[str, list[str], list[list[str]], bool]:
    """Read our own generated pipe-table format into header and data rows."""
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    caption = ""
    if lines and lines[0].startswith("**") and lines[0].endswith("**"):
        caption = lines.pop(0)[2:-2].strip()
    if len(lines) < 2 or not lines[0].startswith("|") or not lines[1].startswith("|"):
        raise ValueError("Converted table is not a valid Markdown pipe table.")

    header = _split_markdown_row(lines[0])
    data_rows: list[list[str]] = []
    for line in lines[2:]:
        if not line.startswith("|"):
            continue
        data_rows.append(_split_markdown_row(line))

    width = max([len(header), *(len(row) for row in data_rows)], default=len(header))
    header += [""] * (width - len(header))
    data_rows = [row + [""] * (width - len(row)) for row in data_rows]
    synthetic_header = bool(header) and all(
        re.fullmatch(r"Column\s+\d+", cell.strip(), re.IGNORECASE)
        for cell in header
    )
    return caption, header, data_rows, not synthetic_header


def _normalize_cell_spacing(value: str) -> str:
    """Repair common OCR word-boundary loss without changing identifiers."""
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = re.sub(r",(?=\S)", ", ", value)
    value = re.sub(r"^developinginstitution$", "Developing institution", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def _join_continuation(left: str, right: str) -> str:
    left = _normalize_cell_spacing(left)
    right = _normalize_cell_spacing(right)
    if not left:
        return right
    if not right:
        return left
    # OCR fragments are concatenated with a readable word boundary. Preserve
    # the source punctuation; this routine does not rewrite scientific names.
    return f"{left.rstrip()} {right.lstrip()}"


def _is_continuation_fragment(table: _ConvertedTable) -> bool:
    """Conservative signal: synthetic header and first row lacks variety cell."""
    if table.has_real_header or table.caption or not table.rows:
        return False
    first = list(table.rows[0])
    if len(first) < 3:
        return False
    # In this corpus, continuation fragments begin with a row whose Variety
    # cell is empty while the Special-features cell continues the prior row.
    return not first[1].strip() and bool(first[2].strip())


def _can_join_table_fragments(
    previous: _ConvertedTable,
    current: _ConvertedTable,
    gap: str,
) -> bool:
    if gap.strip():
        return False
    if previous.page_number is None or current.page_number is None:
        return False
    if current.page_number != previous.page_number + 1:
        return False
    if len(previous.header) != len(current.header):
        return False
    return _is_continuation_fragment(current)


def _merge_table_group(group: list[_ConvertedTable]) -> str:
    """Reconstruct one logical table from adjacent page-level fragments.

    Rules are deliberately narrow:
    * The first fragment supplies the real column headers.
    * The first row of a continuation fragment appends to the previous row
      when its Variety cell is empty and its Special-features cell is present.
    * Blank institution cells on subsequent variety rows inherit the latest
      explicit institution. The original row remains a separate variety row.
    * No row is silently dropped when its structure is ambiguous.
    """
    first = group[0]
    header = [_normalize_cell_spacing(value) for value in first.header]
    rows: list[list[str]] = []
    active_institution = ""

    def add_normal_row(source_row: tuple[str, ...] | list[str]) -> None:
        nonlocal active_institution
        row = [_normalize_cell_spacing(value) for value in source_row]
        row += [""] * (len(header) - len(row))
        if row[0].strip():
            active_institution = row[0]
        elif row[1].strip() and active_institution:
            row[0] = active_institution
        rows.append(row)

    for row in first.rows:
        add_normal_row(row)

    for fragment in group[1:]:
        fragment_rows = [list(row) for row in fragment.rows]
        if not fragment_rows:
            continue

        first_row = [_normalize_cell_spacing(value) for value in fragment_rows[0]]
        first_row += [""] * (len(header) - len(first_row))

        # Reattach a row split across pages. For the Okra table, page 4 adds
        # "Agricultural University." to the institution and continues Kiran's
        # description; page 5 continues Punjab Padmini's description.
        if rows and len(first_row) >= 3 and not first_row[1] and first_row[2]:
            previous_row = rows[-1]
            if first_row[0]:
                previous_row[0] = _join_continuation(previous_row[0], first_row[0])
                active_institution = previous_row[0]
            previous_row[2] = _join_continuation(previous_row[2], first_row[2])
            fragment_rows = fragment_rows[1:]

        for row in fragment_rows:
            add_normal_row(row)

    lines = [
        "| " + " | ".join(value.replace("|", r"\|") for value in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        padded = row + [""] * (len(header) - len(row))
        lines.append(
            "| " + " | ".join(value.replace("|", r"\|") for value in padded) + " |"
        )
    if first.caption:
        lines.insert(0, f"**{first.caption}**")
    return "\n".join(lines)


def convert_html_tables_to_markdown(
    text: str,
    page_boundaries: list[dict[str, int]] | None = None,
) -> HTMLTableConversionResult:
    """Normalize HTML tables and conservatively reconstruct adjacent continuations.

    Cross-page reconstruction is attempted only when explicit PDF page
    boundaries are supplied. Adjacent fragments must be on consecutive pages,
    separated only by whitespace, have equal column counts, and begin with a
    continuation-shaped row (the variety cell is empty).

    Without page boundaries this performs Stage-A representation conversion
    only, and never guesses that two separate tables are related.
    """
    spans = list(_iter_table_spans(text))
    if not spans:
        return HTMLTableConversionResult(text, 0, 0, 0)

    candidates: list[_ConvertedTable | None] = []
    for start, end, table_html in spans:
        markdown = html_table_to_markdown(table_html)
        if markdown is None:
            candidates.append(None)
            continue
        caption, header, rows, has_real_header = _parse_converted_table(markdown)
        candidates.append(
            _ConvertedTable(
                source_start=start,
                source_end=end,
                markdown=markdown,
                page_number=_page_for_offset(start, page_boundaries),
                header=tuple(header),
                rows=tuple(tuple(row) for row in rows),
                has_real_header=has_real_header,
                caption=caption,
            )
        )

    output_parts: list[str] = []
    replacements: list[TableSourceSpan] = []
    cursor = 0
    output_length = 0
    converted = 0
    skipped = 0
    index = 0

    while index < len(spans):
        start, end, table_html = spans[index]
        candidate = candidates[index]

        unchanged = text[cursor:start]
        output_parts.append(unchanged)
        output_length += len(unchanged)

        if candidate is None:
            # Preserve unconvertible HTML verbatim. The caller will fail closed
            # because tables_skipped is nonzero.
            output_parts.append(table_html)
            output_length += len(table_html)
            skipped += 1
            cursor = end
            index += 1
            continue

        group = [candidate]
        last_index = index
        if page_boundaries and candidate.has_real_header:
            while last_index + 1 < len(spans):
                next_index = last_index + 1
                next_candidate = candidates[next_index]
                if next_candidate is None:
                    break
                gap = text[spans[last_index][1] : spans[next_index][0]]
                if not _can_join_table_fragments(group[-1], next_candidate, gap):
                    break
                group.append(next_candidate)
                last_index = next_index

        source_start = group[0].source_start
        source_end = group[-1].source_end
        if len(group) > 1:
            table_markdown = _merge_table_group(group)
        else:
            table_markdown = group[0].markdown

        source_pages = _source_pages_for_range(
            source_start, source_end, page_boundaries
        )
        markdown_start = output_length
        output_parts.append(table_markdown)
        output_length += len(table_markdown)
        replacements.append(
            TableSourceSpan(
                source_start=source_start,
                source_end=source_end,
                markdown_start=markdown_start,
                markdown_end=output_length,
                source_pages=source_pages,
            )
        )
        converted += len(group)
        cursor = source_end
        index = last_index + 1

    output_parts.append(text[cursor:])
    return HTMLTableConversionResult(
        markdown="".join(output_parts),
        tables_found=len(spans),
        tables_converted=converted,
        tables_skipped=skipped,
        replacements=tuple(replacements),
    )

