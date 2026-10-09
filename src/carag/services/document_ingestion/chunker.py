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
    """Count tokens in text using the tokenizer."""
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
            tokenizer=self.tokenizer)

    def chunk(self, markdown: str, metadata: dict[str, Any]) -> list[str]:
        # First parse tables and text with basechef
        tables = self.base_chef.prepare_tables(markdown)
        code = self.base_chef.prepare_code(markdown)
        images = self.base_chef.extract_images(markdown)
        chunks = self.base_chef.extract_chunks(markdown, tables, code, images)

        # Get the specialized case of tables from specialized chef
        parsed_items = self.custom_table_chef.parse(markdown)
        contextual_table_chunks = parsed_items.tables

        chunk_flat = [chunk.text for chunk in chunks]
        code_flat = [cd.content for cd in code]
        image_flat = [img.content for img in images]  # TO BE DISCARDED FOR NOW
        table_flat = [tbl.content for tbl in contextual_table_chunks]

        final_chunks = []

        for chunk_text in chunk_flat:
            if (
                chunk_text.strip()
                and count_tokens(self.tokenizer, chunk_text)
                > settings.CHUNKER_CHUNK_SIZE // settings.CHUNK_TEXT_DIVIDER_FACTOR
            ):
                text_subchunks = self.text_chunker.chunk(chunk_text)
                final_chunks.extend(text_subchunks)
            elif chunk_text.strip():
                final_chunks.append(chunk_text)

        for code_content in code_flat:
            if (
                code_content.strip()
                and count_tokens(self.tokenizer, code_content)
                > settings.CHUNKER_CHUNK_SIZE // settings.CHUNK_TEXT_DIVIDER_FACTOR
            ):
                code_subchunks = self.text_chunker.chunk(code_content)
                final_chunks.extend(code_subchunks)

            elif code_content.strip():
                final_chunks.append(code_content)

        # DISCARD images (usually small, might not need splitting)
        # for image_content in image_flat:
        #     if image_content.strip():
        #         # Images are typically short descriptions
        #         final_chunks.append(image_content)

        # Process contextual tables with table splitter
        for table_content in table_flat:
            if table_content.strip():
                table_subchunks = self.table_chunker.chunk(table_content)
                final_chunks.extend(table_subchunks)

        final_texts: list[str] = []
        for item in final_chunks:
            if isinstance(item, Chunk):
                final_texts.append(item.text)
            elif isinstance(item, str):
                final_texts.append(item)
            else:
                final_texts.append(str(item))

        return final_texts


class EnhancedChunker:
    """Enhanced chunker that optionally applies agentic tagging to chunks."""

    def __init__(self):
        self.settings = get_settings()
        self.base_chunker = CustomChunker()  # Direct instance, no factory call

        # Initialize tagger only if enabled
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
        self, markdown: str, metadata: Optional[dict[str, Any]] = None
    ) -> list[str]:
        """Chunk markdown content with optional agentic tagging."""
        # First, use the base chunker to create chunks
        chunks = self.base_chunker.chunk(markdown, metadata)

        # Apply agentic tagging if enabled and tagger is available
        if self.settings.USE_AGENTIC_TAGGER and self.tagger:
            chunks = self._apply_agentic_tagging(chunks, metadata or {})

        return chunks

    def _apply_agentic_tagging(
        self, chunks: list[str], metadata: dict[str, Any]
    ) -> list[str]:
        """Apply agentic tagging to chunks."""
        try:
            # Create LangChain Document objects for the tagger
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

            # Use the tagger to process documents
            tagged_documents = self.tagger.split_documents(documents)

            # Convert back to strings with enhanced metadata
            enhanced_chunks = []
            for doc in tagged_documents:
                # Append tags as metadata comments
                if doc.metadata.get("tags"):
                    tags_section = f"\n\n<!-- TAGS: {doc.metadata['tags']} -->"
                    enhanced_content = doc.page_content + tags_section
                else:
                    enhanced_content = doc.page_content

                enhanced_chunks.append(enhanced_content)

            logger.info(
                f"Applied agentic tagging to {len(enhanced_chunks)} chunks")
            return enhanced_chunks

        except Exception as e:
            logger.error(
                f"Agentic tagging failed, falling back to base chunks: {e}")
            return chunks  # Fallback to original chunks


# Simple factory function
def get_chunker():
    """Returns the appropriate chunker based on configuration."""
    settings = get_settings()

    if settings.USE_AGENTIC_TAGGER:
        return EnhancedChunker()
    else:
        return CustomChunker()


# Keep your existing function
def get_chunker_original():
    return CustomChunker()
