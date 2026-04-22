"""
VoxHealth — FastAPI 后端 v0.2 (C端完整版)

API端点：
- v1路由：完整C端API（用户/检测/记录/趋势/统计）
- 兼容路由：/api/analyze, /api/diseases, /api/health
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
    title="VoxHealth 🔊",
    description="语音生物标志物AI平台 — 30秒语音，25种疾病早筛",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-User-Id"],
)

# 注册 v1 API 路由
app.include_router(v1_router)

# 前端目录
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
extractor = FeatureExtractor(sr=16000)
detector = DiseaseDetector()


# ── 兼容路由 ──

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>VoxHealth API v0.2 Running</h1>")


@app.get("/api/health")
async def health_check_compat():
    return {
        "status": "healthy", "service": "VoxHealth", "version": "0.2.0",
        "diseases_count": len(DISEASE_REGISTRY),
        "features": ["MFCC", "Jitter/Shimmer", "HNR", "Formants", "Prosody"],
    }


@app.get("/api/diseases")
async def list_diseases_compat():
    categories = {}
    for did, info in DISEASE_REGISTRY.items():
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "id": did, "name": info["name"],
            "markers": info["markers"], "description": info["description"],
        })
    return {"total": len(DISEASE_REGISTRY), "categories": categories}


@app.post("/api/analyze")
async def analyze_audio_compat(audio: UploadFile = File(...)):
    """兼容旧版分析接口"""
    if not audio.filename:
        raise HTTPException(400, "请上传音频文件")
    suffix = Path(audio.filename).suffix.lower()
    if suffix not in (".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"):
        raise HTTPException(400, f"不支持的格式: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        features = extractor.extract(tmp_path)
        report = detector.generate_report(features)
        os.unlink(tmp_path)
        return {
            "report_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "report": report.to_dict(),
            "features": features.to_dict(),
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"分析失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8100))
    print(f"🔊 VoxHealth C端启动于 http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
