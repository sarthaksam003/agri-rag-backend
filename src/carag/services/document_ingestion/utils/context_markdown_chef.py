import re
from typing import List, Tuple

from chonkie.chef.markdown import MarkdownChef
from chonkie.tokenizer import Tokenizer
from chonkie.types import (
    MarkdownDocument,
    MarkdownTable,
)


class ContextualMarkdownChef(MarkdownChef):
    def __init__(
        self,
        tokenizer: Tokenizer | str = "character",
        proximity_window: int = 500,
    ) -> None:
        super().__init__(tokenizer)

        # Improved header pattern to capture both ATX and numbered headers
        self.header_pattern = re.compile(
            r"^(#{1,6}\s+.*|(?:\d+\.)+\s+.*)$", re.MULTILINE
        )

        self.blacklisted_headers = {
            "table of contents",
            "contents",
            "index",
            "list of tables",
            "list of figures",
        }
        self.proximity_window = proximity_window

    def is_blacklisted_header(self, header_text: str) -> bool:
        normalized = header_text.lower().strip()
        return any(
            blacklisted in normalized for blacklisted in self.blacklisted_headers
        )

    def get_header_level(self, header_text: str) -> int:
        """Get the level of a header. Lower number means higher level."""
        if header_text.startswith("#"):
            # ATX style header: count the number of # characters
            level = 0
            for char in header_text:
                if char == "#":
                    level += 1
                else:
                    break
            return level
        else:
            # Numbered header: count the number of dot-separated parts
            match = re.match(r"^((?:\d+\.)+)", header_text)
            if match:
                numbering = match.group(1)
                return numbering.count(".")
            return 1  # Fallback: treat as level 1

    def prepare_headers(self, markdown: str) -> List[Tuple[int, int, str, int]]:
        """Prepare headers with their positions and levels."""
        headers = []
        for match in self.header_pattern.finditer(markdown):
            header_text = match.group(0).strip()
            if self.is_blacklisted_header(header_text):
                continue

            start, end = match.start(), match.end()
            level = self.get_header_level(header_text)
            headers.append((start, end, header_text, level))

        # Sort by position to maintain document order
        headers.sort(key=lambda x: x[0])
        return headers

    def find_relevant_headers_for_table(
        self, table_start: int, all_headers: List[Tuple[int, int, str, int]]
    ) -> List[Tuple[int, int, str, int]]:
        """Find the relevant headers for a table based on hierarchical structure."""
        # Find all headers that come before the table
        preceding_headers = [h for h in all_headers if h[0] < table_start]
        if not preceding_headers:
            return []

        # Build complete section hierarchy
        relevant_headers = []
        current_level = float("inf")  # Start with highest possible level

        # Work backwards to build the full section path
        for header in reversed(preceding_headers):
            if header[3] <= current_level:  # This header is at same or higher level
                relevant_headers.insert(0, header)
                current_level = header[3]

        return relevant_headers

    def normalize_header_text(self, header_text: str) -> str:
        """Normalize header text to ensure consistent formatting."""
        # Remove trailing page numbers like "3" in "3. Detailed Audit Controls 3"
        header_text = re.sub(r"\s+\d+$", "", header_text)

        # Ensure proper formatting for numbered headers
        if not header_text.startswith("#"):
            # For numbered headers like "3.1.2. Information Classification"
            # Ensure they have proper formatting
            match = re.match(r"^((?:\d+\.)+)\s*(.*)", header_text)
            if match:
                numbering, text = match.groups()
                # Remove any extra spaces and normalize
                header_text = f"{numbering.strip()} {text.strip()}"

        return header_text.strip()

    def parse(self, text: str) -> MarkdownDocument:
        """Parse markdown text and return a MarkdownDocument with contextual tables."""
        original_tables = self.prepare_tables(text)
        code_blocks = self.prepare_code(text)
        images = self.extract_images(text)

        # Prepare all non-blacklisted headers with their levels
        all_headers = self.prepare_headers(text)

        modified_tables = []

        for table in original_tables:
            # Find relevant headers for this table
            relevant_headers = self.find_relevant_headers_for_table(
                table.start_index, all_headers
            )

            if relevant_headers:
                # Strategy 1: Build context from header texts only
                header_texts = []
                for header in relevant_headers:
                    normalized_text = self.normalize_header_text(header[2])
                    header_texts.append(normalized_text)

                context = "\n".join(header_texts)
                new_content = (
                    f"{context}\n\n{table.content}" if context else table.content
                )
            else:
                # Fallback: No headers found, use proximity window
                context_start = max(0, table.start_index - self.proximity_window)
                context_text = text[context_start : table.start_index].strip()

                # Clean the context text of any blacklisted headers
                lines = context_text.split("\n")
                clean_lines = []
                for line in lines:
                    line_stripped = line.strip()
                    if self.header_pattern.match(
                        line_stripped
                    ) and self.is_blacklisted_header(line_stripped):
                        continue
                    clean_lines.append(line)
                context_text = "\n".join(clean_lines)

                new_content = (
                    f"{context_text}\n\n{table.content}"
                    if context_text
                    else table.content
                )

            modified_tables.append(
                MarkdownTable(
                    content=new_content,
                    start_index=table.start_index,
                    end_index=table.end_index,
                )
            )

        # Extract chunks using original tables
        # FIXME: Doesnt extract chunks properly, for now, use chonkie's original markdown chef and use this only for tables.
        chunks = self.extract_chunks(text, original_tables, code_blocks, images)

        return MarkdownDocument(
            content=text,
            tables=modified_tables,
            code=code_blocks,
            images=images,
            chunks=chunks,
        )
