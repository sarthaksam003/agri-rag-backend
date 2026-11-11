from typing import Any, Optional

from chonkie import RecursiveChunker, TableChunker
from chonkie.chef import MarkdownChef
from chonkie.chunker import BaseChunker
from chonkie.types import Chunk, RecursiveLevel, RecursiveRules
from langchain_core.documents import Document
from loguru import logger
from tokenizers import Tokenizer

from carag.core.config import get_settings
from carag.core.constants import get_constants
from carag.services.document_ingestion.utils import ContextualMarkdownChef
from carag.services.document_ingestion.utils.html_table_processor import (
    SOURCE_RANGE_MARKER_RE,
    convert_html_tables_to_markdown,
    map_normalized_offset_to_source,
)

settings = get_settings()


chunker_rules: RecursiveRules = RecursiveRules(
    [
        RecursiveLevel(delimiters=["\n# "], include_delim="next"),
        RecursiveLevel(delimiters=["\n## "], include_delim="next"),
        RecursiveLevel(delimiters=["\n### "], include_delim="next"),
        RecursiveLevel(delimiters=["\n#### "], include_delim="next"),
        RecursiveLevel(delimiters=["\n<table>"], include_delim="next"),
        RecursiveLevel(delimiters=["\n"], include_delim="next"),
    ]
)


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    """Count tokens in text using the configured tokenizer."""
    if not text or not text.strip():
        return 0
    return len(tokenizer.encode(text).tokens)


class CustomChunker:
    def __init__(self) -> None:
        self.tokenizer: Tokenizer = Tokenizer.from_pretrained(
            settings.CHUNKER_TOKENIZER_MODEL
        )
        self.tokenizer.no_truncation()
        self.table_chunker: BaseChunker = TableChunker(
            tokenizer="character",
            chunk_size=settings.CHUNKER_CHUNK_SIZE
            * settings.TABLE_LEN_CHUNK_SIZE_FACTOR,
        )
        self.text_chunker: BaseChunker = RecursiveChunker(
            tokenizer_or_token_counter=self.tokenizer,
            chunk_size=settings.CHUNKER_CHUNK_SIZE,
            rules=chunker_rules,
            min_characters_per_chunk=100,
        )
        self.base_chef = MarkdownChef(tokenizer=self.tokenizer)
        self.custom_table_chef = ContextualMarkdownChef(
            tokenizer=self.tokenizer
        )

    @staticmethod
    def _chunk_text_value(item: Any) -> str:
        if isinstance(item, Chunk):
            return item.text
        if isinstance(item, str):
            return item
        return str(item)

    def _chunk_plain_text_segment(self, text: str) -> list[str]:
        """Chunk one text interval that does not contain a parsed table."""
        if not text or not text.strip():
            return []

        code = self.base_chef.prepare_code(text)
        images = self.base_chef.extract_images(text)
        chunks = self.base_chef.extract_chunks(text, [], code, images)

        result: list[str] = []
        threshold = settings.CHUNKER_CHUNK_SIZE // max(
            settings.CHUNK_TEXT_DIVIDER_FACTOR, 1
        )

        for item in chunks:
            chunk_text = self._chunk_text_value(item)
            if not chunk_text.strip():
                continue
            if count_tokens(self.tokenizer, chunk_text) > threshold:
                result.extend(
                    self._chunk_text_value(part)
                    for part in self.text_chunker.chunk(chunk_text)
                )
            else:
                result.append(chunk_text)

        for code_item in code:
            code_content = code_item.content
            if not code_content.strip():
                continue
            if count_tokens(self.tokenizer, code_content) > threshold:
                result.extend(
                    self._chunk_text_value(part)
                    for part in self.text_chunker.chunk(code_content)
                )
            else:
                result.append(code_content)

        # Keep the existing behaviour: image descriptions are not indexed yet.
        return result

    @staticmethod
    def _get_table_context(contextual_content: str, original_content: str) -> str:
        """Separate the heading/proximity context from the actual table body."""
        contextual = contextual_content.strip()
        original = original_content.strip()
        if not contextual or not original or not contextual.endswith(original):
            return ""
        return contextual[: -len(original)].strip()

    def chunk(
        self,
        markdown: str,
        metadata: dict[str, Any],
        page_boundaries: list[dict[str, int]] | None = None,
    ) -> list[str]:
        """Chunk Markdown and MinerU HTML tables while preserving document order.

        MinerU emits HTML tables. They are normalized to Markdown before
        Chonkie's Markdown table parser runs. Plain-text intervals and table
        intervals are then chunked in source order rather than collecting all
        prose first and all tables afterwards.
        """
        conversion = convert_html_tables_to_markdown(markdown, page_boundaries)
        normalized_markdown = conversion.markdown

        if conversion.tables_skipped:
            raise RuntimeError(
                f"Could not convert {conversion.tables_skipped} HTML table(s) to Markdown. "
                "Refusing to send unparsed HTML tables through ordinary text chunking."
            )

        if conversion.tables_found:
            logger.info(
                "Normalized MinerU HTML tables: found={}, source_fragments_converted={}, "
                "logical_table_groups={}, skipped={}",
                conversion.tables_found,
                conversion.tables_converted,
                len(conversion.replacements),
                conversion.tables_skipped,
            )

        parsed_document = self.custom_table_chef.parse(normalized_markdown)
        contextual_tables = parsed_document.tables
        original_tables = self.base_chef.prepare_tables(normalized_markdown)

        # Guard against parser disagreement. If a table was normalized but is
        # not recognized by Chonkie, fail clearly rather than silently treating
        # raw table syntax as ordinary prose.
        if len(contextual_tables) != len(original_tables):
            raise RuntimeError(
                "Chonkie table parsers disagreed: "
                f"contextual={len(contextual_tables)}, base={len(original_tables)}."
            )

        # ``tables_converted`` counts the original HTML fragments. Stage B may
        # intentionally reconstruct several adjacent page fragments into one
        # logical Markdown table, so the correct expected count is the number
        # of replacement groups, not the number of source fragments.
        expected_logical_tables = len(conversion.replacements)
        if len(original_tables) < expected_logical_tables:
            raise RuntimeError(
                "Chonkie recognized fewer Markdown tables than the HTML converter "
                f"produced logical table groups ({len(original_tables)} < "
                f"{expected_logical_tables}; source HTML fragments="
                f"{conversion.tables_converted}). Refusing to silently chunk "
                "unrecognized tables as prose."
            )

        final_chunks: list[str] = []
        cursor = 0

        # MarkdownTable start/end indexes refer to positions in the normalized
        # source text. Process each interval and table in document order.
        ordered_tables = sorted(
            zip(original_tables, contextual_tables),
            key=lambda pair: pair[0].start_index,
        )

        for original_table, contextual_table in ordered_tables:
            start = int(original_table.start_index)
            end = int(original_table.end_index)

            if start < cursor or end <= start or end > len(normalized_markdown):
                raise RuntimeError(
                    "Chonkie returned invalid or overlapping table offsets: "
                    f"start={start}, end={end}, cursor={cursor}, "
                    f"document_length={len(normalized_markdown)}."
                )

            # Preserve and chunk prose that appears before this table.
            final_chunks.extend(
                self._chunk_plain_text_segment(normalized_markdown[cursor:start])
            )

            table_text = normalized_markdown[start:end].strip()
            if not table_text:
                table_text = original_table.content.strip()
            context = self._get_table_context(
                contextual_table.content,
                original_table.content,
            )

            source_span = next(
                (
                    replacement
                    for replacement in conversion.replacements
                    if replacement.markdown_start <= start < replacement.markdown_end
                ),
                None,
            )
            source_start = (
                source_span.source_start
                if source_span
                else map_normalized_offset_to_source(conversion, start)
            )
            source_end = (
                source_span.source_end
                if source_span
                else map_normalized_offset_to_source(conversion, end)
            )
            source_pages = source_span.source_pages if source_span else ()
            page_suffix = (
                ":PAGES=" + ",".join(str(page) for page in source_pages)
                if source_pages
                else ""
            )
            source_marker = (
                f"\n<!-- AGRICHAT_SOURCE_RANGE:{source_start}:{source_end}"
                f"{page_suffix} -->"
            )

            table_parts = self.table_chunker.chunk(table_text)
            if not table_parts:
                logger.warning(
                    "Table at character range {}–{} produced no chunks; "
                    "preserving its source Markdown as one chunk.",
                    start,
                    end,
                )
                table_chunks = [table_text]
            else:
                table_chunks = [self._chunk_text_value(part) for part in table_parts]

            for table_chunk in table_chunks:
                if not table_chunk.strip():
                    continue
                if context:
                    final_chunks.append(
                        f"{context}\n\n{table_chunk}{source_marker}"
                    )
                else:
                    final_chunks.append(f"{table_chunk}{source_marker}")

            cursor = end

        # Process the remaining text after the final table.
        final_chunks.extend(
            self._chunk_plain_text_segment(normalized_markdown[cursor:])
        )

        return [chunk for chunk in final_chunks if chunk and chunk.strip()]


class EnhancedChunker:
    """Enhanced chunker that optionally applies agentic tagging to chunks."""

    def __init__(self):
        self.settings = get_settings()
        self.base_chunker = CustomChunker()

        self.tagger = None
        if self.settings.USE_AGENTIC_TAGGER:
            try:
                from carag.services.document_ingestion.tagger import AgenticTagger

                constants = get_constants()
                self.tagger = AgenticTagger(llm=constants.llm)
                logger.info("Agentic Tagger initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Agentic Tagger: {e}")

    def chunk(
        self,
        markdown: str,
        metadata: Optional[dict[str, Any]] = None,
        page_boundaries: list[dict[str, int]] | None = None,
    ) -> list[str]:
        """Chunk markdown content with optional agentic tagging."""
        chunks = self.base_chunker.chunk(
            markdown, metadata or {}, page_boundaries=page_boundaries
        )

        if self.settings.USE_AGENTIC_TAGGER and self.tagger:
            chunks = self._apply_agentic_tagging(chunks, metadata or {})

        return chunks

    def _apply_agentic_tagging(
        self, chunks: list[str], metadata: dict[str, Any]
    ) -> list[str]:
        """Apply agentic tagging to chunks."""
        try:
            documents = []
            for i, chunk_text in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_index": i,
                        "original_chunk_length": len(chunk_text),
                    }
                )
                documents.append(
                    Document(page_content=chunk_text, metadata=chunk_metadata)
                )

            tagged_documents = self.tagger.split_documents(documents)
            enhanced_chunks = []
            for doc in tagged_documents:
                if doc.metadata.get("tags"):
                    tags_section = f"\n\n<!-- TAGS: {doc.metadata['tags']} -->"
                    enhanced_content = doc.page_content + tags_section
                else:
                    enhanced_content = doc.page_content
                enhanced_chunks.append(enhanced_content)

            logger.info(f"Applied agentic tagging to {len(enhanced_chunks)} chunks")
            return enhanced_chunks

        except Exception as e:
            logger.error(
                f"Agentic tagging failed, falling back to base chunks: {e}"
            )
            return chunks


def get_chunker():
    """Return the configured chunker implementation."""
    settings = get_settings()
    if settings.USE_AGENTIC_TAGGER:
        return EnhancedChunker()
    return CustomChunker()


def get_chunker_original():
    return CustomChunker()
