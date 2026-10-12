"""FastAPI application: upload documents, then chat within their scope."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.core.engine import ChatEngine, LLMUnavailableError
from app.core.extract import SUPPORTED_EXTENSIONS, ExtractionError
from app.core.persona import PRESETS, Persona, PersonaStore
from app.core.store import DocumentStore, DuplicateDocumentError
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentModel,
    PersonaModel,
    PresetRequest,
    StatusResponse,
    UploadResponse,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

settings = get_settings()
store = DocumentStore(
    data_dir=settings.data_dir,
    chunk_size=settings.chunk_size,
    overlap=settings.chunk_overlap,
)
personas = PersonaStore(settings.data_dir)
engine = ChatEngine(settings=settings, store=store)

app = FastAPI(
    title="Scoped Document Chatbot Engine",
    description=(
        "Upload documents, and the chatbot answers questions within that subject "
        "area only — refusing anything outside it. Configure the persona to turn "
        "the same engine into a customer-care agent, a course TA, or anything else."
    ),
    version="1.0.0",
)


@app.exception_handler(LLMUnavailableError)
async def _llm_unavailable_handler(_request, exc: LLMUnavailableError):
    # 503: the service is fine, its upstream dependency is not.
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------


@app.get("/api/status", response_model=StatusResponse, tags=["status"])
def status() -> StatusResponse:
    """Health and configuration summary. Safe to call without an API key."""
    persona = personas.current
    return StatusResponse(
        status="ok",
        llm_configured=settings.has_api_key,
        model=settings.model,
        documents=len(store.list_documents()),
        chunks=store.total_chunks,
        persona=PersonaModel(**persona.to_dict()),
        greeting=persona.effective_greeting(),
        strict_scope=settings.strict_scope,
        relevance_floor=settings.relevance_floor,
    )


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


@app.post("/api/documents", response_model=UploadResponse, tags=["documents"])
async def upload_documents(files: List[UploadFile] = File(...)) -> UploadResponse:
    """Ingest one or more documents into the knowledge base.

    Partial success is intentional: one bad file in a batch should not discard
    the good ones. Failures are reported per-file in ``skipped``.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    uploaded, skipped = [], []

    for upload in files:
        filename = upload.filename or "unnamed"
        try:
            data = await upload.read()
        except Exception as exc:
            skipped.append({"name": filename, "reason": "Could not read: {}".format(exc)})
            continue
        finally:
            await upload.close()

        if len(data) > settings.max_upload_bytes:
            skipped.append(
                {
                    "name": filename,
                    "reason": "File is {:.1f} MB; the limit is {} MB.".format(
                        len(data) / 1024 / 1024, settings.max_upload_mb
                    ),
                }
            )
            continue

        try:
            doc = store.add(filename, data)
            uploaded.append(DocumentModel(**doc.to_dict()))
        except DuplicateDocumentError as exc:
            skipped.append({"name": filename, "reason": str(exc)})
        except ExtractionError as exc:
            skipped.append({"name": filename, "reason": str(exc)})
        except Exception as exc:
            logger.exception("Failed to ingest %s", filename)
            skipped.append({"name": filename, "reason": "Unexpected error: {}".format(exc)})

    if not uploaded and skipped:
        # Nothing landed — surface it as an error rather than a misleading 200.
        raise HTTPException(
            status_code=400,
            detail="No files could be ingested. " + "; ".join(
                "{}: {}".format(s["name"], s["reason"]) for s in skipped
            ),
        )

    return UploadResponse(
        uploaded=uploaded,
        skipped=skipped,
        total_documents=len(store.list_documents()),
        total_chunks=store.total_chunks,
    )


@app.get("/api/documents", response_model=List[DocumentModel], tags=["documents"])
def list_documents() -> List[DocumentModel]:
    """List everything currently in the knowledge base."""
    return [DocumentModel(**d.to_dict()) for d in store.list_documents()]


@app.delete("/api/documents/{doc_id}", tags=["documents"])
def delete_document(doc_id: str):
    """Remove a single document and de-index its chunks."""
    if not store.delete(doc_id):
        raise HTTPException(status_code=404, detail="No document with id '{}'.".format(doc_id))
    return {
        "deleted": doc_id,
        "total_documents": len(store.list_documents()),
        "total_chunks": store.total_chunks,
    }


@app.delete("/api/documents", tags=["documents"])
def clear_documents():
    """Wipe the entire knowledge base."""
    store.clear()
    return {"cleared": True, "total_documents": 0, "total_chunks": 0}


@app.get("/api/supported-types", tags=["documents"])
def supported_types():
    return {"extensions": sorted(SUPPORTED_EXTENSIONS)}


# --------------------------------------------------------------------------
# Persona
# --------------------------------------------------------------------------


@app.get("/api/persona", response_model=PersonaModel, tags=["persona"])
def get_persona() -> PersonaModel:
    return PersonaModel(**personas.current.to_dict())


@app.put("/api/persona", response_model=PersonaModel, tags=["persona"])
def set_persona(body: PersonaModel) -> PersonaModel:
    """Define who the bot is and what subject it is allowed to cover."""
    saved = personas.set(Persona.from_dict(body.model_dump()))
    return PersonaModel(**saved.to_dict())


@app.post("/api/persona/preset", response_model=PersonaModel, tags=["persona"])
def apply_preset(body: PresetRequest) -> PersonaModel:
    """Switch to a built-in persona (customer_care, course_ta, generic)."""
    try:
        saved = personas.apply_preset(body.preset)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return PersonaModel(**saved.to_dict())


@app.get("/api/persona/presets", tags=["persona"])
def list_presets():
    return {key: value.to_dict() for key, value in PRESETS.items()}


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
def chat(body: ChatRequest) -> ChatResponse:
    """Ask a question. Off-topic questions are refused without an LLM call."""
    result = engine.answer(
        question=body.message,
        persona=personas.current,
        history=[turn.model_dump() for turn in body.history],
        doc_ids=body.doc_ids,
    )
    return ChatResponse(**result.to_dict())


# --------------------------------------------------------------------------
# Web UI
# --------------------------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(WEB_DIR / "index.html"))
