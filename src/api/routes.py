"""
VoiceHealth — C端完整API路由

用户系统 + 健康记录 + 趋势分析 + AI解读
"""

import os
import sys
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from src.core.feature_extractor import FeatureExtractor
from src.core.disease_detector import DiseaseDetector, DISEASE_REGISTRY
from src.core.database import db
from src.core.ai_insight import insight_generator
from src.core.voice_verifier import (
    LivenessDetector, ReadingVerifier, SpeechRecognizer, READING_TEXTS
)

router = APIRouter(prefix="/api/v1", tags=["VoiceHealth API v1"])

extractor = FeatureExtractor(sr=16000)
detector = DiseaseDetector()
liveness_detector = LivenessDetector(sr=16000)
reading_verifier = ReadingVerifier()
speech_recognizer = SpeechRecognizer()

# ── Pydantic Models ──

class UserRegister(BaseModel):
    phone: str
    nickname: str = ""
    gender: str = ""
    age: int = 0

class UserLogin(BaseModel):
    phone: str

class UserProfile(BaseModel):
    nickname: str = ""
    gender: str = ""
    age: int = 0
    health_goals: list = []

# ── 简单认证 ──

def get_user_id(x_user_id: Optional[str] = Header(None)):
    """从Header获取用户ID（简化版认证）"""
    if not x_user_id:
        raise HTTPException(401, "请先登录")
    user = db.get_user(x_user_id)
    if not user:
        raise HTTPException(401, "用户不存在")
    return x_user_id


# ═══════ 用户系统 ═══════

@router.post("/user/register")
async def register(data: UserRegister):
    """用户注册（手机号）"""
    user = db.create_user(data.phone, data.nickname, data.gender, data.age)
    return {"ok": True, "user": user}

@router.post("/user/login")
async def login(data: UserLogin):
    """用户登录"""
    user = db.get_user_by_phone(data.phone)
    if not user:
        # 自动注册
        user = db.create_user(data.phone)
    return {"ok": True, "user": user}

@router.get("/user/profile")
async def get_profile(user_id: str = Depends(get_user_id)):
    """获取用户档案"""
    user = db.get_user(user_id)
    stats = db.get_stats(user_id)
    return {"ok": True, "user": user, "stats": stats}

@router.put("/user/profile")
async def update_profile(data: UserProfile, user_id: str = Depends(get_user_id)):
    """更新用户档案"""
    db.update_user(user_id, **data.dict())
    user = db.get_user(user_id)
    return {"ok": True, "user": user}


# ═══════ 语音检测 ═══════

@router.post("/analyze")
async def analyze_audio(audio: UploadFile = File(...), x_user_id: Optional[str] = Header(None)):
    """
    语音健康检测 — 核心接口

    上传30秒语音，返回健康报告 + AI解读
    """
    suffix = Path(audio.filename or "audio.wav").suffix.lower()
    if suffix not in (".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"):
        raise HTTPException(400, f"不支持的格式: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 特征提取
        features = extractor.extract(tmp_path)
        # 疾病检测
        report = detector.generate_report(features)
        report_dict = report.to_dict()
        features_dict = features.to_dict()

        # AI解读
        user_name = "您"
        if x_user_id:
            user = db.get_user(x_user_id)
            if user:
                user_name = user.get("nickname", "您")

        ai_insight = await insight_generator.generate_insight(report_dict, user_name)

        # 保存记录
        record_id = None
        if x_user_id:
            record_id = db.save_health_record(
                x_user_id, report_dict, features_dict,
                ai_insight=ai_insight, audio_path=""
            )

        # 清理音频（隐私）
        os.unlink(tmp_path)

        return {
            "ok": True,
            "record_id": record_id,
            "timestamp": datetime.now().isoformat(),
            "report": report_dict,
            "features": features_dict,
            "ai_insight": ai_insight,
        }

    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"分析失败: {str(e)}")


# ═══════ 健康记录 ═══════

@router.get("/records")
async def get_records(user_id: str = Depends(get_user_id), limit: int = 20):
    """获取检测记录列表"""
    records = db.get_user_records(user_id, limit)
    return {"ok": True, "records": records, "total": len(records)}

@router.get("/records/{record_id}")
async def get_record(record_id: str, user_id: str = Depends(get_user_id)):
    """获取单条记录详情"""
    records = db.get_user_records(user_id, limit=100)
    for r in records:
        if r["id"] == record_id:
            return {"ok": True, "record": r}
    raise HTTPException(404, "记录不存在")


# ═══════ 趋势分析 ═══════

@router.get("/trends")
async def get_trends(user_id: str = Depends(get_user_id), days: int = 30):
    """获取健康趋势数据"""
    trends = db.get_trends(user_id, days)
    trend_summary = await insight_generator.generate_trend_summary(
        trends, db.get_user(user_id).get("nickname", "您")
    ) if trends else "暂无趋势数据"
    return {"ok": True, "trends": trends, "summary": trend_summary}

@router.get("/stats")
async def get_stats(user_id: str = Depends(get_user_id)):
    """获取用户统计"""
    stats = db.get_stats(user_id)
    return {"ok": True, "stats": stats}


# ═══════ 疾病信息 ═══════

@router.get("/diseases")
async def list_diseases():
    """获取可检测疾病列表"""
    categories = {}
    for did, info in DISEASE_REGISTRY.items():
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "id": did, "name": info["name"],
            "markers": info["markers"], "description": info["description"],
        })
    return {"ok": True, "total": len(DISEASE_REGISTRY), "categories": categories}


# ═══════ 健康检查 ═══════

@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "VoiceHealth",
        "version": "0.2.0",
        "diseases": len(DISEASE_REGISTRY),
    }


# ═══════ 语音验证 ═══════

@router.get("/verification/text")
async def get_reading_text():
    """获取随机朗读文本"""
    text = reading_verifier.get_random_text()
    return {"ok": True, "text": text}


@router.get("/verification/texts")
async def list_reading_texts():
    """获取所有朗读文本"""
    return {"ok": True, "texts": READING_TEXTS}


@router.post("/verification/liveness")
async def check_liveness(audio: UploadFile = File(...)):
    """
    活体检测 - 检查是否为真人发声
    
    检测维度：
    1. 背景噪声分析
    2. 频谱连续性
    3. 动态范围
    4. 微变化分析
    5. 谐波结构
    """
    suffix = Path(audio.filename or "audio.wav").suffix.lower()
    if suffix not in (".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"):
        raise HTTPException(400, f"不支持的格式: {suffix}")
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = liveness_detector.detect(tmp_path)
        os.unlink(tmp_path)
        
        return {
            "ok": True,
            "is_live": result.is_live,
            "confidence": result.confidence,
            "score": result.score,
            "checks": result.checks,
            "details": result.details,
            "reason": result.reason
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"检测失败: {str(e)}")


@router.post("/verification/reading")
async def verify_reading(
    audio: UploadFile = File(...),
    expected_text_id: str = "standard_1"
):
    """
    朗读验证 - 检查是否正确朗读指定文本
    
    流程：
    1. 语音识别（ASR）
    2. 与预设文本比对
    3. 计算匹配度
    """
    suffix = Path(audio.filename or "audio.wav").suffix.lower()
    if suffix not in (".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"):
        raise HTTPException(400, f"不支持的格式: {suffix}")
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # 语音识别
        recognized_text = speech_recognizer.recognize(tmp_path)
        
        # 验证朗读
        result = reading_verifier.verify(recognized_text, expected_text_id)
        
        os.unlink(tmp_path)
        
        return {
            "ok": True,
            "is_valid": result.is_valid,
            "confidence": result.confidence,
            "recognized_text": result.recognized_text,
            "expected_text": result.expected_text,
            "match_ratio": result.match_ratio,
            "word_count": result.word_count,
            "expected_word_count": result.expected_word_count,
            "reason": result.reason
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"验证失败: {str(e)}")


@router.post("/verification/full")
async def full_verification(
    audio: UploadFile = File(...),
    expected_text_id: str = "standard_1"
):
    """
    完整验证 - 活体检测 + 朗读验证
    
    这是推荐的验证方式，同时检查：
    1. 是否为真人发声
    2. 是否正确朗读指定文本
    """
    suffix = Path(audio.filename or "audio.wav").suffix.lower()
    if suffix not in (".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"):
        raise HTTPException(400, f"不支持的格式: {suffix}")
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # 活体检测
        liveness_result = liveness_detector.detect(tmp_path)
        
        # 语音识别
        recognized_text = speech_recognizer.recognize(tmp_path)
        
        # 朗读验证
        reading_result = reading_verifier.verify(recognized_text, expected_text_id)
        
        os.unlink(tmp_path)
        
        # 综合判断
        is_valid = liveness_result.is_live and reading_result.is_valid
        
        return {
            "ok": True,
            "is_valid": is_valid,
            "liveness": {
                "is_live": liveness_result.is_live,
                "confidence": liveness_result.confidence,
                "score": liveness_result.score,
                "checks": liveness_result.checks,
                "reason": liveness_result.reason
            },
            "reading": {
                "is_valid": reading_result.is_valid,
                "confidence": reading_result.confidence,
                "recognized_text": reading_result.recognized_text,
                "match_ratio": reading_result.match_ratio,
                "reason": reading_result.reason
            }
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"验证失败: {str(e)}")


# ═══════ 视频分析 ═══════

@router.post("/video/analyze")
async def analyze_video_api(
    video: UploadFile = File(...),
    x_user_id: Optional[str] = Header(None)
):
    """
    视频健康分析 - 皮肤/眼睛/头发状态检测
    
    上传视频，返回：
    - 皮肤状态（肤色、痘痘、皱纹、色斑、毛孔）
    - 眼睛状态（黑眼圈、眼袋、红血丝、疲劳度）
    - 头发状态（发量、发质、白发比例）
    - 综合评分和生物学年龄预测
    """
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in (".mp4", ".avi", ".mov", ".webm", ".mkv"):
        raise HTTPException(400, f"不支持的视频格式: {suffix}")
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # 分析视频
        result = analyze_video(tmp_path)
        
        # 保存记录
        record_id = None
        if x_user_id:
            record_id = db.save_video_record(
                x_user_id, result.to_dict()
            )
        
        os.unlink(tmp_path)
        
        return {
            "ok": True,
            "record_id": record_id,
            "result": result.to_dict()
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"视频分析失败: {str(e)}")


@router.post("/video/skin")
async def analyze_skin_api(
    video: UploadFile = File(...),
    x_user_id: Optional[str] = Header(None)
):
    """皮肤状态分析"""
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in (".mp4", ".avi", ".mov", ".webm", ".mkv"):
        raise HTTPException(400, f"不支持的视频格式: {suffix}")
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = analyze_video(tmp_path)
        os.unlink(tmp_path)
        
        return {
            "ok": True,
            "skin": result.skin.to_dict()
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"皮肤分析失败: {str(e)}")


@router.post("/video/eye")
async def analyze_eye_api(
    video: UploadFile = File(...),
    x_user_id: Optional[str] = Header(None)
):
    """眼睛状态分析"""
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in (".mp4", ".avi", ".mov", ".webm", ".mkv"):
        raise HTTPException(400, f"不支持的视频格式: {suffix}")
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = analyze_video(tmp_path)
        os.unlink(tmp_path)
        
        return {
            "ok": True,
            "eye": result.eye.to_dict()
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"眼睛分析失败: {str(e)}")


@router.post("/video/hair")
async def analyze_hair_api(
    video: UploadFile = File(...),
    x_user_id: Optional[str] = Header(None)
):
    """头发状态分析"""
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in (".mp4", ".avi", ".mov", ".webm", ".mkv"):
        raise HTTPException(400, f"不支持的视频格式: {suffix}")
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = analyze_video(tmp_path)
        os.unlink(tmp_path)
        
        return {
            "ok": True,
            "hair": result.hair.to_dict()
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"头发分析失败: {str(e)}")
