import os
from typing import Any

from carag.core.constants import get_constants
from carag.services.document_ingestion.ingest import get_ingestion_engine
from carag.worker import celery_app

constants = get_constants()


@celery_app.task(bind=True, name="ingest_document")
def ingest_document(
    self,
    tenant_id: str,
    filename: str,
    file_path: str,
    meta: dict[str, Any],
) -> dict[str, Any]:

    document_id = meta["document_id"]
    keep_file = False

    try:
        ingestion_engine = get_ingestion_engine(tenant_id)

        ingested = ingestion_engine.ingest_single_document(
            file_path,
            meta,
        )

        keep_file = True

        return {
            "status": "completed",
            "document_id": document_id,
            "filename": filename,
            "chunks_processed": len(ingested.chunks),
            "processing_time": ingested.processing_time,
        }

    except FileExistsError as exc:
        return {
            "status": "skipped",
            "document_id": document_id,
            "filename": filename,
            "error_message": str(exc),
        }

    except Exception as exc:
        return {
            "status": "failed",
            "document_id": document_id,
            "filename": filename,
            "error_message": str(exc),
        }

    finally:
        if not keep_file:
            try:
                os.unlink(file_path)
            except OSError:
                pass