"""
Demo FastAPI backend for testing the frontend without PyTorch.
This simulates the image processing without actually running the model.
"""

import os
import uuid
import time
import asyncio
from pathlib import Path
from typing import Optional
from threading import Thread

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="NAFNet Deblurring (Demo)", version="1.0.0")

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
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Processing jobs storage
jobs = {}


class JobStatus(BaseModel):
    id: str
    status: str
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
    """Get system status - Demo mode"""
    return {
        "gpu_available": True,
        "gpu_name": "Demo Mode (No GPU)",
        "vram_gb": 8.0,
        "models": [
            {"name": "nafnet32", "weights_available": True, "recommended_vram": 8},
            {"name": "nafnet64", "weights_available": True, "recommended_vram": 24},
            {"name": "nafnet_deblur", "weights_available": True, "recommended_vram": 40}
        ],
        "tile_size": 512
    }


@app.post("/api/restore")
async def restore_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_type: str = "nafnet32"
):
    """Upload and restore an image (demo mode - just copies the image)"""

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
    background_tasks.add_task(process_job_demo, job_id)

    return {"job_id": job_id, "status": "pending"}


def process_job_demo(job_id: str):
    """Demo processing - simulates the processing steps"""
    job = jobs[job_id]
    job.status = "processing"

    start_time = time.time()

    # Simulate processing steps
    steps = [
        (10, 0.3),
        (20, 0.3),
        (40, 0.5),
        (60, 0.5),
        (80, 0.5),
        (90, 0.3),
        (100, 0.2)
    ]

    for progress, delay in steps:
        time.sleep(delay)
        job.progress = progress

    # In demo mode, just copy the input as output
    import shutil
    shutil.copy(job.input_path, job.output_path)

    job.status = "completed"
    job.processing_time = round(time.time() - start_time, 2)


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

    for path in [job.input_path, job.output_path]:
        if path and Path(path).exists():
            Path(path).unlink()

    del jobs[job_id]
    return {"status": "deleted"}


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("  Sharp Restore - Demo Mode")
    print("  Open http://localhost:8000 in your browser")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
