"""
VoxHealth — FastAPI 后端

API端点：
- POST /api/analyze — 上传音频，返回健康报告
- GET  /api/diseases — 获取可检测疾病列表
- GET  /api/health — 健康检查
"""

import os
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

# 确保可以导入 core 模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.core.feature_extractor import FeatureExtractor
from src.core.disease_detector import DiseaseDetector, DISEASE_REGISTRY

app = FastAPI(
    title="VoxHealth 🔊",
    description="语音生物标志物AI平台 — 30秒语音，25种疾病早筛",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化引擎
extractor = FeatureExtractor(sr=16000)
detector = DiseaseDetector()

# 前端目录
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>VoxHealth API Running</h1>")


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "VoxHealth",
        "version": "0.1.0",
        "diseases_count": len(DISEASE_REGISTRY),
        "features": ["MFCC", "Jitter/Shimmer", "HNR", "Formants", "Prosody"],
    }


@app.get("/api/diseases")
async def list_diseases():
    """获取可检测疾病列表"""
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
    return {
        "total": len(DISEASE_REGISTRY),
        "categories": categories,
    }


@app.post("/api/analyze")
async def analyze_audio(audio: UploadFile = File(...)):
    """
    分析语音音频

    上传 WAV/MP3/OGG 格式的30秒语音，
    返回完整的健康筛查报告。
    """
    # 验证文件
    if not audio.filename:
        raise HTTPException(400, "请上传音频文件")

    suffix = Path(audio.filename).suffix.lower()
    if suffix not in (".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"):
        raise HTTPException(400, f"不支持的格式: {suffix}，请上传 WAV/MP3/OGG")

    # 保存临时文件
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 提取声学特征
        features = extractor.extract(tmp_path)

        # 生成健康报告
        report = detector.generate_report(features)

        # 清理音频文件（隐私保护）
        os.unlink(tmp_path)

        return {
            "report_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "report": report.to_dict(),
            "features": features.to_dict(),
        }

    except Exception as e:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"分析失败: {str(e)}")


# ── 启动 ──

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8100))
    print(f"🔊 VoxHealth 启动于 http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
