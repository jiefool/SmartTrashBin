"""
FastAPI application – exposes REST endpoints for bin status, alerts, and
trash classification.
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger

from app.config import settings
from app.models.bin_status import BinReading
from app.models.classification import ClassificationResult
from app.sensors.ultrasonic import SimulatedUltrasonicSensor
from app.services.camera_service import camera
from app.services.classifier_service import classifier
from app.services.monitor_service import MonitorService
from app.utils.logging import setup_logging

monitor = MonitorService()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    setup_logging()
    logger.info(f"Starting {settings.app_name} ...")

    # Register simulated bins (replace with real sensors in production)
    for bin_id, fill_pct in [("BIN-A", 0.0), ("BIN-B", 50.0), ("BIN-C", 75.0)]:
        monitor.register_sensor(
            SimulatedUltrasonicSensor(
                bin_id=bin_id,
                capacity_cm=settings.bin_capacity_cm,
                initial_fill_percent=fill_pct,
            )
        )

    monitor.start()

    # Load the trash classification model (non-blocking if missing)
    classifier.load_model()

    # Initialise camera (non-blocking if no camera attached)
    camera.initialise()

    yield
    monitor.stop()
    camera.release()
    logger.info(f"{settings.app_name} shut down.")


app = FastAPI(
    title=settings.app_name,
    description="IoT Smart Trash Bin monitoring system",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["System"])
async def root() -> dict[str, Any]:
    return {
        "app": settings.app_name,
        "version": "1.0.0",
        "classifier_ready": classifier.is_ready,
        "camera_ready": camera.is_ready,
        "endpoints": {
            "health": "/health",
            "all_bins": "/bins",
            "single_bin": "/bins/{bin_id}",
            "classify": "POST /classify",
            "capture_and_classify": "POST /capture-and-classify",
            "docs": "/docs",
        },
    }


@app.get("/health", tags=["System"])
async def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.get("/bins", response_model=dict[str, Any], tags=["Bins"])
async def list_bins() -> dict[str, Any]:
    """Return the latest reading for every registered bin."""
    readings = monitor.get_latest_readings()
    return {
        bin_id: {
            "fill_level_percent": r.fill_level_percent,
            "fill_status": r.fill_status,
            "distance_cm": r.distance_cm,
            "capacity_cm": r.capacity_cm,
            "timestamp": r.timestamp.isoformat(),
        }
        for bin_id, r in readings.items()
    }


@app.get("/bins/{bin_id}", response_model=dict[str, Any], tags=["Bins"])
async def get_bin(bin_id: str) -> dict[str, Any]:
    """Return the latest reading for a specific bin."""
    readings = monitor.get_latest_readings()
    if bin_id not in readings:
        raise HTTPException(status_code=404, detail=f"Bin '{bin_id}' not found.")
    r: BinReading = readings[bin_id]
    return {
        "bin_id": r.bin_id,
        "fill_level_percent": r.fill_level_percent,
        "fill_status": r.fill_status,
        "distance_cm": r.distance_cm,
        "capacity_cm": r.capacity_cm,
        "timestamp": r.timestamp.isoformat(),
    }



@app.post("/classify", response_model=ClassificationResult, tags=["Classification"])
async def classify_image(file: UploadFile = File(...)) -> ClassificationResult:
    """
    Upload an image of trash and get its predicted category.

    Categories (TrashNet): cardboard, glass, metal, paper, plastic, trash.
    """
    if not classifier.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Classification model not loaded. "
                "Train it first: python -m app.training.train"
            ),
        )

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got content type: '{content_type}'",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = classifier.classify(image_bytes)
    return result


@app.post("/capture-and-classify", response_model=ClassificationResult, tags=["Classification"])
async def capture_and_classify() -> ClassificationResult:
    """
    Capture an image from the attached camera and classify the trash.

    No file upload needed – the Pi camera takes the photo automatically.
    Categories (TrashNet): cardboard, glass, metal, paper, plastic, trash.
    """
    if not camera.is_ready:
        raise HTTPException(
            status_code=503,
            detail="No camera available. Attach a camera and restart the app.",
        )

    if not classifier.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Classification model not loaded. "
                "Train it first: python -m app.training.train"
            ),
        )

    # Capture image from camera
    image_bytes = camera.capture()

    # Save the captured image for reference
    saved_path = camera.save_capture(image_bytes)
    logger.info(f"Captured image saved to '{saved_path}'")

    # Classify
    result = classifier.classify(image_bytes)
    return result