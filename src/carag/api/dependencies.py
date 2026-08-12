from functools import lru_cache

from fastapi import Header, HTTPException

from carag.core.config import get_settings

settings = get_settings()


@lru_cache
def get_tenant_id(x_tenant_id: str = Header("demo-tenant")) -> str:
    """
    Dependency to get tenant ID from header.
    For now, defaults to 'demo-tenant' until JWT auth is implemented.
    """
    # TODO: Add JWT validation later
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="Tenant ID header is required")

    return x_tenant_id


def get_ingestion_engine(tenant_id: str):
    """Dependency to get ingestion engine for a tenant."""
    from carag.services.document_ingestion.ingest import get_ingestion_engine

    return get_ingestion_engine(tenant_id=tenant_id)
