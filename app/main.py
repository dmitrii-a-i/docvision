"""FastAPI application for document processing."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.pipeline import Pipeline
from app.schemas import HealthResponse, ProcessResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Set HF cache before any model imports
os.environ.setdefault("HF_HOME", settings.HF_HOME)

pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    log.info("Starting up — loading models...")
    pipeline = Pipeline()
    log.info("Models loaded, ready to serve.")
    yield
    log.info("Shutting down.")


app = FastAPI(title="DocVision API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process", response_model=ProcessResponse)
async def process_document(file: UploadFile):
    """Process a document image: dewarp, detect fields, extract text."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        result = pipeline.process(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if pipeline is None:
        return HealthResponse(
            status="loading", device=settings.device, models_loaded=[],
        )
    return HealthResponse(
        status="ok",
        device=settings.device,
        models_loaded=pipeline.models_loaded,
    )
