"""
FastAPI backend for NAFNet image deblurring.
"""

import os
import sys
import uuid
import time
import asyncio
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from nafnet import nafnet_width32, nafnet_width64, nafnet_deblur
from inference import process_image

app = FastAPI(title="NAFNet Deblurring", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
WEIGHTS_DIR = Path("weights")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)

# Global model cache
_model_cache = {}
_device = None


def get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def get_model(model_type: str):
    """Load and cache model"""
    if model_type in _model_cache:
        return _model_cache[model_type]

    device = get_device()

    if model_type == "nafnet32":
        model = nafnet_width32()
    elif model_type == "nafnet64":
        model = nafnet_width64()
    else:
        model = nafnet_deblur()

    # Try to load weights
    weights_path = WEIGHTS_DIR / f"{model_type}.pth"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    _model_cache[model_type] = model

    return model


def get_tile_size():
    """Auto-determine tile size based on VRAM"""
    if not torch.cuda.is_available():
        return 256

    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram < 8:
        return 256
    elif vram < 16:
        return 512
    elif vram < 24:
        return 768
    return None


# Processing jobs storage
jobs = {}


class JobStatus(BaseModel):
    id: str
    status: str  # pending, processing, completed, failed
    progress: int
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/status")
async def get_status():
    """Get system status"""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if gpu_available else 0

    # Check for available weights
    available_models = []
    for model_type in ["nafnet32", "nafnet64", "nafnet_deblur"]:
        weights_path = WEIGHTS_DIR / f"{model_type}.pth"
        available_models.append({
            "name": model_type,
            "weights_available": weights_path.exists(),
            "recommended_vram": {"nafnet32": 8, "nafnet64": 24, "nafnet_deblur": 40}[model_type]
        })

    return {
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "vram_gb": round(vram, 1),
        "models": available_models,
        "tile_size": get_tile_size()
    }


@app.post("/api/restore")
async def restore_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_type: str = "nafnet32"
):
    """Upload and restore an image"""

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"Invalid file type: {file.content_type}")

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Save uploaded file
    ext = Path(file.filename).suffix or ".jpg"
    input_path = UPLOAD_DIR / f"{job_id}_input{ext}"
    output_path = OUTPUT_DIR / f"{job_id}_output{ext}"

    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    # Create job
    jobs[job_id] = JobStatus(
        id=job_id,
        status="pending",
        progress=0,
        input_path=str(input_path),
        output_path=str(output_path)
    )

    # Start processing in background
    background_tasks.add_task(process_job, job_id, model_type)

    return {"job_id": job_id, "status": "pending"}


def process_job(job_id: str, model_type: str):
    """Process image in background"""
    job = jobs[job_id]
    job.status = "processing"
    job.progress = 10

    try:
        start_time = time.time()

        # Load image
        img = cv2.imread(job.input_path)
        if img is None:
            raise ValueError("Could not read image")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        job.progress = 20

        # Load model
        model = get_model(model_type)
        device = get_device()
        tile_size = get_tile_size()
        job.progress = 40

        # Process
        result = process_image(img, model, device, tile_size)
        job.progress = 90

        # Save result
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(job.output_path, result_bgr)

        job.status = "completed"
        job.progress = 100
        job.processing_time = round(time.time() - start_time, 2)

    except Exception as e:
        job.status = "failed"
        job.error = str(e)


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/image/{image_type}/{job_id}")
async def get_image(image_type: str, job_id: str):
    """Get input or output image"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]

    if image_type == "input":
        path = job.input_path
    elif image_type == "output":
        path = job.output_path
    else:
        raise HTTPException(400, "Invalid image type")

    if not path or not Path(path).exists():
        raise HTTPException(404, "Image not found")

    return FileResponse(path)


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]

    # Delete files
    for path in [job.input_path, job.output_path]:
        if path and Path(path).exists():
            Path(path).unlink()

    del jobs[job_id]
    return {"status": "deleted"}


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
