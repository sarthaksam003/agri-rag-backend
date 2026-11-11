"""Local MinerU pipeline conversion for AgriChat.

This implementation uses MinerU's local ``pipeline`` backend. It does not
connect to an HTTP VLM server and is suitable for CPU-only local development.
Page-range parameters are 0-based and inclusive, matching MinerU's do_parse API.
"""

from __future__ import annotations
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from threading import RLock
import gc
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from mineru.cli.common import do_parse
from mineru.utils.enum_class import MakeMode

from carag.core.config import get_settings


def _ensure_stdout_encoding() -> None:
    """Work around Windows console wrappers without replacing normal stdout."""
    stdout = sys.stdout
    if getattr(stdout, "encoding", None):
        return

    buffer = getattr(stdout, "buffer", None)
    if buffer is None:
        return

    try:
        sys.stdout = io.TextIOWrapper(
            buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    except (AttributeError, OSError, ValueError):
        # MinerU may still work if stdout cannot be wrapped.
        logger.debug(
            "Could not apply UTF-8 stdout workaround on this console.")


# MinerU's merge_table reference is a module-level global.
# Protect the temporary override while MinerU is processing a PDF.
_MINERU_PIPELINE_LOCK = RLock()


@contextmanager
def _mineru_table_merge_control(enabled: bool) -> Iterator[None]:
    """
    Control MinerU's cross-page table merging.

    MinerU 2.6.7 calls merge_table() unconditionally in:
    mineru.backend.pipeline.model_json_to_middle_json

    Its MINERU_TABLE_MERGE_ENABLE environment variable is not consulted
    by that version. Therefore, when merging is disabled, temporarily
    replace the pipeline module's imported function with a no-op.

    Restore the original function even if PDF parsing raises an exception.
    """

    with _MINERU_PIPELINE_LOCK:
        if enabled:
            logger.info("MinerU cross-page table merging is enabled.")
            yield
            return

        pipeline_module = import_module(
            "mineru.backend.pipeline.model_json_to_middle_json"
        )

        original_merge_table = getattr(
            pipeline_module,
            "merge_table",
            None,
        )

        if not callable(original_merge_table):
            raise RuntimeError(
                "Could not locate MinerU's pipeline merge_table function. "
                "The installed MinerU version may have changed."
            )

        def _skip_cross_page_table_merge(
            _page_info_list: Any,
        ) -> None:
            """Preserve individual page tables for AgriChat's Stage B."""
            logger.info(
                "AgriChat intercepted MinerU merge_table(): "
                "skipping automatic cross-page merging for {} pages.",
                len(_page_info_list) if _page_info_list else 0,
            )
            return None

        pipeline_module.merge_table = _skip_cross_page_table_merge

        logger.info(
            "MinerU cross-page table merging is disabled. "
            "Individual page tables will be preserved for AgriChat Stage B."
        )

        try:
            yield
        finally:
            pipeline_module.merge_table = original_merge_table

            logger.debug(
                "Restored MinerU's original merge_table function."
            )


def _extract_table_html_from_page_info(page_info: dict[str, Any]) -> list[str]:
    """Recover MinerU table HTML when its Markdown renderer omits a table-only page.

    Only ``para_blocks`` are traversed. ``preproc_blocks`` are intentionally
    excluded because they can duplicate the same OCR/table representation.
    The HTML is taken verbatim from MinerU's structured output; this function
    does not infer or reconstruct any table text.
    """
    recovered: list[str] = []

    def visit(node: Any, inside_table: bool = False) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            is_table_context = inside_table or node_type in {
                "table", "table_body"}
            html_value = node.get("html")

            if (
                is_table_context
                and isinstance(html_value, str)
                and "<table" in html_value.lower()
            ):
                html_value = html_value.strip()
                if html_value:
                    recovered.append(html_value)
                return

            for child in node.values():
                visit(child, is_table_context)

        elif isinstance(node, list):
            for child in node:
                visit(child, inside_table)

    for block in page_info.get("para_blocks") or []:
        visit(block)

    return recovered


def parse_doc_to_markdown(
    path_list: list[Path],
    output_dir: str,
    server_url: str | None = None,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    *,
    write_markdown: bool = False,
    language: str = "en",
) -> list[dict[str, Any]]:
    """Parse PDFs locally with MinerU's pipeline backend and return page Markdown.

    ``server_url`` remains as an ignored compatibility parameter for old callers
    that previously supplied ``PDFMD_API_BASE``. The local pipeline does not use
    it. New callers should omit it.

    ``start_page_id`` and ``end_page_id`` are 0-based, inclusive page indexes.
    ``end_page_id=None`` means through the final page.

    MinerU writes a middle JSON file to disk. We load that structured output and
    render page-level Markdown, avoiding the need to keep a second full-document
    Markdown string in memory unless ``write_markdown=True`` is requested.
    """
    if not path_list:
        raise ValueError("At least one PDF path must be provided.")
    if start_page_id < 0:
        raise ValueError(
            "start_page_id must be greater than or equal to zero.")
    if end_page_id is not None and end_page_id < start_page_id:
        raise ValueError(
            "end_page_id must be greater than or equal to start_page_id.")
    if not language or not language.strip():
        raise ValueError("MinerU language must be a non-empty string.")

    if server_url:
        logger.warning(
            "Ignoring server_url={!r}: local MinerU pipeline backend is selected.",
            server_url,
        )

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    settings = get_settings()

    # MinerU reads this flag directly from the process environment. Pydantic
    # loads values from .env but does not export them to os.environ.
    table_merge_enabled = settings.MINERU_TABLE_MERGE_ENABLE
    os.environ["MINERU_TABLE_MERGE_ENABLE"] = (
        "1" if table_merge_enabled else "0"
    )
    logger.info(
        "MinerU cross-page table merging is {}.",
        "enabled" if table_merge_enabled else "disabled",
    )

    all_page_markdowns: list[dict[str, Any]] = []

    for input_path in path_list:
        pdf_path = Path(input_path).expanduser().resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are supported: {pdf_path}")

        # MinerU's output naming convention uses the supplied file name as the
        # task name. A per-call TemporaryDirectory prevents collisions in the
        # normal ingestion path.
        pdf_file_name = pdf_path.name
        middle_json_path = (
            output_root
            / pdf_file_name
            / "auto"
            / f"{pdf_file_name}_middle.json"
        )

        logger.info(
            "Starting local MinerU pipeline for {} (pages {}–{}).",
            pdf_file_name,
            start_page_id,
            "end" if end_page_id is None else end_page_id,
        )
        _ensure_stdout_encoding()
        stage_start = time.perf_counter()

        pdf_bytes = pdf_path.read_bytes()
        try:
            with _mineru_table_merge_control(table_merge_enabled):
                do_parse(
                    output_dir=str(output_root),
                    pdf_file_names=[pdf_file_name],
                    pdf_bytes_list=[pdf_bytes],
                    p_lang_list=[language.strip()],
                    backend="pipeline",
                    parse_method="auto",
                    formula_enable=True,
                    table_enable=True,
                    server_url=None,
                    f_draw_layout_bbox=False,
                    f_draw_span_bbox=False,
                    f_dump_md=False,
                    f_dump_middle_json=True,
                    f_dump_model_output=False,
                    f_dump_orig_pdf=False,
                    f_dump_content_list=False,
                    f_make_md_mode=MakeMode.MM_MD,
                    start_page_id=start_page_id,
                    end_page_id=end_page_id,
                )
        finally:
            # Release the original PDF bytes as soon as MinerU returns/fails.
            del pdf_bytes

        logger.info(
            "MinerU pipeline analysis completed for {} in {:.2f}s.",
            pdf_file_name,
            time.perf_counter() - stage_start,
        )

        if not middle_json_path.is_file():
            raise FileNotFoundError(
                "MinerU completed without creating the expected middle JSON: "
                f"{middle_json_path}"
            )

        with middle_json_path.open("r", encoding="utf-8") as middle_file:
            middle_json = json.load(middle_file)

        pdf_info = middle_json.get("pdf_info") if isinstance(
            middle_json, dict) else None
        if not isinstance(pdf_info, list) or not pdf_info:
            raise RuntimeError(
                f"MinerU produced no page information for {pdf_file_name}."
            )

        # Validate page identity before rendering. In batch mode MinerU's
        # page_idx values are relative to the selected page range.
        page_indexes: list[int] = []
        for page_info in pdf_info:
            if not isinstance(page_info, dict) or page_info.get("page_idx") is None:
                raise RuntimeError(
                    f"MinerU returned a page record without page_idx for {pdf_file_name}."
                )
            try:
                page_indexes.append(int(page_info["page_idx"]))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"MinerU returned an invalid page_idx for {pdf_file_name}: "
                    f"{page_info.get('page_idx')!r}."
                ) from exc

        if len(page_indexes) != len(set(page_indexes)):
            raise RuntimeError(
                f"MinerU returned duplicate page indexes for {pdf_file_name}: "
                f"{page_indexes}."
            )

        if end_page_id is not None:
            expected_local_indexes = set(
                range(end_page_id - start_page_id + 1))
            actual_local_indexes = set(page_indexes)
            missing_indexes = sorted(
                expected_local_indexes - actual_local_indexes)
            unexpected_indexes = sorted(
                actual_local_indexes - expected_local_indexes)
            if missing_indexes or unexpected_indexes:
                raise RuntimeError(
                    f"MinerU page coverage mismatch for {pdf_file_name}, "
                    f"requested PDF pages {start_page_id + 1}–{end_page_id + 1}: "
                    f"missing local indexes={missing_indexes}, "
                    f"unexpected local indexes={unexpected_indexes}."
                )

        # LLM-assisted title leveling is optional. The previous default made
        # an external structured-output request for every document and could
        # spend significant time retrying. Keep it opt-in.
        titled_pdf_info = pdf_info
        if settings.MINERU_LLM_TITLE_ENABLE:
            logger.info(
                "LLM-assisted title leveling is enabled for {}.", pdf_file_name)
            try:
                # Import lazily so disabled title enhancement does not initialize
                # ChatOpenAI or Langfuse.
                from carag.services.document_ingestion.title_maker import Titler

                titled_pdf_info = Titler().llm_aided_title(pdf_info)
                if not isinstance(titled_pdf_info, list) or not titled_pdf_info:
                    raise RuntimeError(
                        "Title post-processing returned no page information."
                    )
            except Exception:
                logger.exception(
                    "LLM-assisted title leveling failed for {}; using deterministic levels.",
                    pdf_file_name,
                )
                titled_pdf_info = pdf_info
        else:
            logger.info(
                "LLM-assisted title leveling is disabled; using deterministic title levels for {}.",
                pdf_file_name,
            )

        # The Markdown renderer expects title blocks to carry a hierarchy level.
        # Keep levels already supplied by MinerU or the optional LLM. When no
        # usable level exists, use level 3 rather than making extraction depend
        # on a remote title-classification service.
        for page_info in titled_pdf_info:
            for block in page_info.get("para_blocks", []):
                if block.get("type") == "title":
                    level = block.get("level")
                    if not isinstance(level, int) or not 0 <= level <= 6:
                        block["level"] = 3

        image_dir = output_root / pdf_file_name / "auto" / "images"
        page_records: dict[int, str] = {}
        empty_markdown_pages: list[int] = []

        for page_info in titled_pdf_info:
            page_idx_value = page_info.get("page_idx")
            if page_idx_value is None:
                # The original page records were validated above. If an optional
                # title pass drops page_idx, fail rather than silently losing a page.
                raise RuntimeError(
                    f"Page metadata lost page_idx during title processing for {pdf_file_name}."
                )

            # MinerU's page-range pipeline returns page_idx relative to the
            # selected subset. Restore the original 1-based PDF page number.
            local_page_idx = int(page_idx_value)
            original_page_number = start_page_id + local_page_idx + 1

            page_markdown = pipeline_union_make(
                [page_info],
                MakeMode.MM_MD,
                str(image_dir),
            )
            markdown_text = str(page_markdown or "").strip()

            # Some table-only pages contain valid OCR HTML in MinerU's middle
            # JSON, but pipeline_union_make() returns an empty string for them.
            # Recover that existing HTML verbatim so downstream HTML table
            # normalization and cross-page reconstruction can process it.
            if not markdown_text:
                recovered_tables = _extract_table_html_from_page_info(
                    page_info)
                if recovered_tables:
                    markdown_text = "\n\n".join(recovered_tables)
                    logger.warning(
                        "MinerU Markdown renderer returned no content for PDF page {} in {}; "
                        "recovered {} raw HTML table(s) from structured output.",
                        original_page_number,
                        pdf_file_name,
                        len(recovered_tables),
                    )

            if not markdown_text:
                empty_markdown_pages.append(original_page_number)
                block_count = len(page_info.get("para_blocks") or [])
                if block_count:
                    logger.warning(
                        "MinerU returned {} parsed block(s) but no Markdown or recoverable table HTML "
                        "for PDF page {} in {}.",
                        block_count,
                        original_page_number,
                        pdf_file_name,
                    )
                else:
                    logger.info(
                        "PDF page {} in {} contains no renderable text blocks; retaining its page record.",
                        original_page_number,
                        pdf_file_name,
                    )
            page_records[original_page_number] = markdown_text

        if end_page_id is not None:
            expected_page_numbers = set(
                range(start_page_id + 1, end_page_id + 2)
            )
            actual_page_numbers = set(page_records)
            missing_pages = sorted(expected_page_numbers - actual_page_numbers)
            if missing_pages:
                raise RuntimeError(
                    f"MinerU did not render page records for PDF page(s) {missing_pages} "
                    f"in {pdf_file_name}."
                )

        if not any(content.strip() for content in page_records.values()):
            raise RuntimeError(
                f"MinerU did not produce usable page Markdown for {pdf_file_name}."
            )

        page_markdowns = [
            {
                "page_number": page_number,
                "content": content,
                "source_file": str(pdf_path),
                "extraction_method": "mineru-pipeline",
                "has_markdown": bool(content.strip()),
            }
            for page_number, content in sorted(page_records.items())
        ]

        if empty_markdown_pages:
            logger.warning(
                "MinerU returned empty Markdown for {} page(s) in {}: {}.",
                len(empty_markdown_pages),
                pdf_file_name,
                empty_markdown_pages,
            )
        all_page_markdowns.extend(page_markdowns)

        if write_markdown:
            complete_markdown = "\n\n".join(
                page["content"] for page in page_markdowns
            )
            markdown_path = output_root / pdf_file_name / \
                "auto" / f"{pdf_path.stem}.md"
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(complete_markdown, encoding="utf-8")
            logger.info("Markdown written to {}.", markdown_path)

        logger.info(
            "MinerU generated Markdown for {} page(s) from {}.",
            len(page_markdowns),
            pdf_file_name,
        )

        # Drop large intermediate structures before moving to another PDF.
        del middle_json, pdf_info, titled_pdf_info
        gc.collect()

    if not all_page_markdowns:
        raise RuntimeError(
            "MinerU produced no page-level Markdown for the supplied PDFs.")

    return all_page_markdowns


if __name__ == "__main__":
    from carag.core.config import get_settings

    cli_settings = get_settings()
    input_dir = Path(cli_settings.PDFMD_DATA_RAW)
    output_dir = str(Path(cli_settings.PDFMD_DATA_PROCESSED))

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    parse_doc_to_markdown(pdf_paths, output_dir, write_markdown=True)
