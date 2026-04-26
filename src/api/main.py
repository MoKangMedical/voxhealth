"""
VoxHealth -- FastAPI Backend v0.3

API Endpoints:
- v1 Router: Full C-end API (user/records/trends/stats)
- Compat Router: /api/analyze, /api/diseases, /api/health
"""

import os
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.core.feature_extractor import FeatureExtractor
from src.core.disease_detector import DiseaseDetector, DISEASE_REGISTRY
from src.api.routes import router as v1_router

app = FastAPI(
    title="VoxHealth",
    description="Voice Biomarker AI Platform -- 30s Voice, 25 Disease Screening",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-User-Id"],
)

# Register v1 API router
app.include_router(v1_router)

# Frontend directory
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
extractor = FeatureExtractor(sr=16000)
detector = DiseaseDetector()


# -- Compat Routes --

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>VoxHealth API v0.3 Running</h1>")


@app.get("/api/health")
async def health_check_compat():
    return {
        "status": "healthy",
        "service": "VoxHealth",
        "version": "0.3.0",
        "diseases": len(DISEASE_REGISTRY),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/analyze")
async def analyze_compat(audio: UploadFile = File(...)):
    """Compat analyze endpoint (no auth required)"""
    from src.api.routes import analyze_audio
    return await analyze_audio(audio)


@app.get("/api/diseases")
async def diseases_compat():
    """Compat diseases endpoint"""
    categories = {}
    for did, info in DISEASE_REGISTRY.items():
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "id": did,
            "name": info["name"],
            "markers": info["markers"],
            "description": info["description"],
        })
    return {"ok": True, "total": len(DISEASE_REGISTRY), "categories": categories}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("VOXHEALTH_PORT", "8100"))
    host = os.getenv("VOXHEALTH_HOST", "0.0.0.0")
    print(f"[VoxHealth] Starting on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
