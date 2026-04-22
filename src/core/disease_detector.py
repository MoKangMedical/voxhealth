"""
VoxHealth — 疾病检测引擎

基于声学特征的多疾病筛查系统。
覆盖5大类别25种疾病/健康状态。

架构设计：
- 每种疾病类别有独立的检测模型
- 支持Mock模式（无训练数据时的降级）
- 输出风险评分 0-100 + 置信度
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from .feature_extractor import AcousticFeatures


@dataclass
class DiseaseRisk:
    """单项疾病风险评估"""
    disease: str
    category: str
    risk_score: float        # 0-100, 越高越需关注
    confidence: float        # 0-1, 检测置信度
    risk_level: str          # 低/中/高
    key_markers: List[str]   # 关键声学标志物
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HealthReport:
    """完整健康报告"""
    overall_risk_level: str
    overall_score: float
    diseases: List[DiseaseRisk]
    feature_summary: Dict[str, str]
    recommendations: List[str]
    recording_quality: str
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "overall_risk_level": self.overall_risk_level,
            "overall_score": self.overall_score,
            "diseases": [d.to_dict() for d in self.diseases],
            "feature_summary": self.feature_summary,
            "recommendations": self.recommendations,
            "recording_quality": self.recording_quality,
            "timestamp": self.timestamp,
        }


# ── 疾病定义 ──

DISEASE_REGISTRY = {
    # ═══ 心理健康 ═══
    "depression": {
        "name": "抑郁风险", "category": "心理健康",
        "markers": ["语速降低", "音调单调", "停顿增多", "能量下降", "动态范围缩小"],
        "description": "语音中的情绪低落信号：语速慢、音调变化小、频繁停顿",
    },
    "anxiety": {
        "name": "焦虑状态", "category": "心理健康",
        "markers": ["语速加快", "音调升高", "Jitter增大", "呼吸急促", "声音颤抖"],
        "description": "焦虑时交感神经兴奋导致的声音变化",
    },
    "burnout": {
        "name": "职业倦怠", "category": "心理健康",
        "markers": ["语速缓慢", "能量极低", "停顿异常", "音域狭窄"],
        "description": "长期压力导致的身心疲惫反映在语音中",
    },
    "stress": {
        "name": "心理压力", "category": "心理健康",
        "markers": ["音调升高", "语速波动", "Shimmer增大", "HNR下降"],
        "description": "压力水平的语音指标",
    },
    "social_isolation": {
        "name": "社交孤立", "category": "心理健康",
        "markers": ["语速减慢", "对话韵律异常", "响应延迟", "音调单一"],
        "description": "长期缺乏社交互动的语音特征",
    },

    # ═══ 认知退行 ═══
    "parkinsons": {
        "name": "帕金森病早期", "category": "认知退行",
        "markers": ["Jitter显著↑", "Shimmer增大", "HNR下降", "语速不稳", "发声震颤"],
        "description": "帕金森影响发声肌群，震颤在运动症状出现前即可在语音中检测",
    },
    "alzheimers": {
        "name": "阿尔茨海默早期", "category": "认知退行",
        "markers": ["词汇检索延迟", "句子不完整", "重复模式", "语速异常", "停顿模式改变"],
        "description": "认知退行影响语言组织能力",
    },
    "mci": {
        "name": "轻度认知障碍", "category": "认知退行",
        "markers": ["停顿增加", "词汇量下降", "句法简化", "F0变异增大"],
        "description": "认知功能轻微下降的早期信号",
    },
    "frailty": {
        "name": "老年衰弱", "category": "认知退行",
        "markers": ["音量下降", "语速极慢", "能量低", "发声无力"],
        "description": "老年衰弱综合征的语音特征",
    },

    # ═══ 呼吸系统 ═══
    "copd": {
        "name": "慢阻肺风险", "category": "呼吸系统",
        "markers": ["呼气缩短", "浊音比例↓", "辅音弱化", "呼吸声明显"],
        "description": "COPD影响呼气功能，导致语音气流异常",
    },
    "asthma": {
        "name": "哮喘风险", "category": "呼吸系统",
        "markers": ["呼吸音异常", "语音中断", "气流不稳"],
        "description": "气道阻塞对语音的影响",
    },
    "respiratory_distress": {
        "name": "呼吸窘迫", "category": "呼吸系统",
        "markers": ["短促语音", "呼吸频率↑", "发声困难"],
        "description": "呼吸困难导致的语音变化",
    },

    # ═══ 心血管 ═══
    "hypertension": {
        "name": "高血压风险", "category": "心血管",
        "markers": ["语速异常", "音调紧张", "呼吸模式改变"],
        "description": "血压异常对神经系统的影响反映在语音中",
    },
    "heart_failure": {
        "name": "心衰风险", "category": "心血管",
        "markers": ["呼吸困难", "语速极慢", "音量微弱", "频繁停顿"],
        "description": "心功能下降导致的全身供血不足影响发声",
    },

    # ═══ 代谢疾病 ═══
    "diabetes_t2": {
        "name": "2型糖尿病风险", "category": "代谢疾病",
        "markers": ["舌运动变化", "发音清晰度↓", "共振峰偏移"],
        "description": "高血糖影响神经和肌肉功能，改变发音特征",
    },
    "thyroid": {
        "name": "甲状腺异常", "category": "代谢疾病",
        "markers": ["音调变化", "语速异常", "声带水肿特征"],
        "description": "甲状腺功能异常影响声带和发音",
    },

    # ═══ 其他健康状态 ═══
    "sleep_quality": {
        "name": "睡眠质量", "category": "健康状态",
        "markers": ["能量水平", "语音疲劳度", "语速稳定性"],
        "description": "睡眠不足反映在语音活力中",
    },
    "fatigue": {
        "name": "疲劳状态", "category": "健康状态",
        "markers": ["RMS下降", "语速减慢", "停顿增多", "音调降低"],
        "description": "身体疲劳对语音的影响",
    },
    "alcohol_influence": {
        "name": "酒精影响", "category": "健康状态",
        "markers": ["发音模糊", "Jitter↑↑", "Shimmer↑", "语速异常", "协调性↓"],
        "description": "酒精影响小脑和运动皮层，导致语音协调性下降",
    },
    "pain_level": {
        "name": "疼痛程度", "category": "健康状态",
        "markers": ["音调紧张", "发声受限", "呼吸抑制", "能量降低"],
        "description": "疼痛导致的发声模式改变",
    },
    "emotional_state": {
        "name": "情绪状态", "category": "健康状态",
        "markers": ["F0变异", "能量动态", "语速模式", "频谱倾斜"],
        "description": "整体情绪状态评估",
    },
    "voice_disorder": {
        "name": "嗓音障碍", "category": "健康状态",
        "markers": ["Jitter/Shimmer异常↑", "HNR严重↓", "发声困难"],
        "description": "声带本身的病理变化",
    },
    "hearing_impairment": {
        "name": "听力下降迹象", "category": "健康状态",
        "markers": ["音量异常大", "音调控制偏差", "共振峰偏移"],
        "description": "听力下降对自我语音监控的影响",
    },
    "cognitive_load": {
        "name": "认知负荷", "category": "健康状态",
        "markers": ["语速下降", "停顿增加", "发音不流畅"],
        "description": "高负荷认知任务对语音流畅性的影响",
    },
    "autism_spectrum": {
        "name": "自闭谱系特征", "category": "健康状态",
        "markers": ["韵律异常", "音调单一", "节奏刻板"],
        "description": "ASD相关的语音韵律特征",
    },
}


class DiseaseDetector:
    """
    疾病检测引擎

    当前为Mock模式 — 使用规则化启发式算法。
    替换为训练好的模型即可实现真实检测。
    """

    def __init__(self, model_dir: str = ""):
        self.model_dir = model_dir
        self.diseases = DISEASE_REGISTRY

    def detect(self, features: AcousticFeatures) -> List[DiseaseRisk]:
        """对一组声学特征进行多疾病检测"""
        risks = []
        for disease_id, info in self.diseases.items():
            risk = self._evaluate_disease(disease_id, info, features)
            risks.append(risk)

        # 按风险评分排序
        risks.sort(key=lambda r: r.risk_score, reverse=True)
        return risks

    def _evaluate_disease(self, disease_id: str, info: dict, feat: AcousticFeatures) -> DiseaseRisk:
        """基于规则的疾病风险评估（Mock模式）"""
        score = 0.0
        markers_found = []

        if disease_id == "depression":
            if feat.speech_rate < 2.5: score += 25; markers_found.append("语速偏低")
            if feat.f0_std < 20: score += 20; markers_found.append("音调单调")
            if feat.pause_ratio > 0.3: score += 20; markers_found.append("停顿较多")
            if feat.rms_mean < 0.02: score += 15; markers_found.append("能量偏低")
            if feat.energy_std < 0.001: score += 10; markers_found.append("动态范围小")

        elif disease_id == "anxiety":
            if feat.speech_rate > 4.5: score += 25; markers_found.append("语速偏快")
            if feat.f0_mean > 200: score += 15; markers_found.append("音调偏高")
            if feat.jitter_local > 0.02: score += 20; markers_found.append("Jitter增大")
            if feat.shimmer_local > 0.05: score += 15; markers_found.append("Shimmer增大")
            if feat.f0_std > 60: score += 10; markers_found.append("音调波动大")

        elif disease_id == "parkinsons":
            if feat.jitter_local > 0.025: score += 30; markers_found.append("Jitter显著↑")
            if feat.shimmer_local > 0.06: score += 25; markers_found.append("Shimmer↑")
            if feat.hnr_mean < 10: score += 20; markers_found.append("HNR↓")
            if feat.speech_rate < 2.0 or feat.speech_rate > 5.0:
                score += 15; markers_found.append("语速不稳")
            if feat.f0_std > 50: score += 10; markers_found.append("发声震颤迹象")

        elif disease_id == "copd":
            if feat.spectral_flatness_mean > 0.1: score += 20; markers_found.append("频谱平坦")
            if feat.pause_ratio > 0.4: score += 25; markers_found.append("呼气缩短")
            if feat.rms_std > 0.03: score += 15; markers_found.append("气流不稳")
            if feat.speech_rate < 2.0: score += 15; markers_found.append("语速缓慢")

        elif disease_id == "fatigue":
            if feat.rms_mean < 0.015: score += 25; markers_found.append("RMS偏低")
            if feat.speech_rate < 2.5: score += 20; markers_found.append("语速缓慢")
            if feat.pause_ratio > 0.35: score += 20; markers_found.append("停顿较多")
            if feat.f0_mean < 100: score += 15; markers_found.append("音调偏低")
            if feat.energy_std < 0.001: score += 10; markers_found.append("能量波动小")

        elif disease_id == "alcohol_influence":
            if feat.jitter_local > 0.03: score += 30; markers_found.append("Jitter↑↑")
            if feat.shimmer_local > 0.08: score += 25; markers_found.append("Shimmer↑↑")
            if feat.spectral_centroid_std > 500: score += 20; markers_found.append("发音不稳定")
            if feat.zero_crossing_rate_mean > 0.15: score += 15; markers_found.append("噪声增多")

        elif disease_id == "stress":
            if feat.f0_mean > 180: score += 20; markers_found.append("音调升高")
            if feat.jitter_local > 0.015: score += 15; markers_found.append("Jitter增大")
            if feat.shimmer_local > 0.04: score += 15; markers_found.append("Shimmer增大")
            if feat.hnr_mean < 12: score += 15; markers_found.append("HNR下降")
            if feat.rms_std > 0.025: score += 10; markers_found.append("能量波动大")

        elif disease_id == "voice_disorder":
            if feat.jitter_local > 0.04: score += 30; markers_found.append("Jitter严重↑")
            if feat.shimmer_local > 0.1: score += 30; markers_found.append("Shimmer严重↑")
            if feat.hnr_mean < 5: score += 25; markers_found.append("HNR严重↓")

        elif disease_id == "frailty":
            if feat.rms_mean < 0.01: score += 30; markers_found.append("音量极低")
            if feat.speech_rate < 1.5: score += 25; markers_found.append("语速极慢")
            if feat.f0_mean < 80: score += 20; markers_found.append("发声无力")
            if feat.pause_ratio > 0.5: score += 15; markers_found.append("停顿极多")

        elif disease_id == "sleep_quality":
            if feat.rms_mean < 0.02: score += 20; markers_found.append("能量不足")
            if feat.speech_rate < 3.0: score += 15; markers_found.append("语速偏慢")
            if feat.f0_std < 25: score += 15; markers_found.append("音调单调")
            if feat.pause_count > 5: score += 10; markers_found.append("频繁停顿")

        elif disease_id == "diabetes_t2":
            if abs(feat.formant_f1 - 500) > 200: score += 15; markers_found.append("F1偏移")
            if feat.spectral_flatness_mean > 0.08: score += 15; markers_found.append("清晰度下降")
            if feat.speech_rate < 2.5: score += 10; markers_found.append("语速偏慢")

        else:
            # 通用启发式：基于特征值异常
            vec = feat.to_vector()
            norm = np.linalg.norm(vec[:10])  # 用前10维做粗估
            if norm > 2.0:
                score += min(30, norm * 5)
                markers_found.append("声学特征偏移")

        # 添加随机扰动模拟模型不确定性
        score = np.clip(score + np.random.normal(0, 3), 0, 100)
        confidence = min(0.85, 0.4 + score / 200)  # Mock置信度

        risk_level = "低" if score < 30 else ("中" if score < 60 else "高")

        return DiseaseRisk(
            disease=info["name"],
            category=info["category"],
            risk_score=round(float(score), 1),
            confidence=round(float(confidence), 2),
            risk_level=risk_level,
            key_markers=markers_found if markers_found else ["未检测到明显异常"],
            description=info["description"],
        )

    def generate_report(self, features: AcousticFeatures) -> HealthReport:
        """生成完整健康报告"""
        diseases = self.detect(features)

        # 总体评分
        high_risks = [d for d in diseases if d.risk_level == "高"]
        medium_risks = [d for d in diseases if d.risk_level == "中"]

        if high_risks:
            overall = "需关注"
            overall_score = max(d.risk_score for d in high_risks)
        elif medium_risks:
            overall = "轻微关注"
            overall_score = max(d.risk_score for d in medium_risks)
        else:
            overall = "正常"
            overall_score = max(d.risk_score for d in diseases) if diseases else 0

        # 特征摘要
        feature_summary = {
            "基频": f"{features.f0_mean:.0f} Hz (变异{features.f0_std:.1f})",
            "Jitter": f"{features.jitter_local:.4f}",
            "Shimmer": f"{features.shimmer_local:.4f}",
            "HNR": f"{features.hnr_mean:.1f} dB",
            "语速": f"{features.speech_rate:.1f} 音节/秒",
            "停顿比": f"{features.pause_ratio:.1%}",
            "RMS能量": f"{features.rms_mean:.4f}",
        }

        # 建议
        recommendations = []
        if high_risks:
            top = high_risks[0]
            recommendations.append(f"⚠️ 【{top.disease}】风险较高（{top.risk_score:.0f}分），建议专业医疗咨询。")
        for d in medium_risks[:2]:
            recommendations.append(f"💡 【{d.disease}】处于中等风险，建议定期关注。")
        recommendations.append("📌 此报告仅供参考，不构成医疗诊断。如有不适请咨询专业医生。")

        # 录音质量评估
        if features.rms_mean < 0.005:
            quality = "信号较弱，建议提高音量重新录制"
        elif features.pause_ratio > 0.6:
            quality = "停顿过多，建议连续说话30秒"
        elif features.speech_rate < 1.0:
            quality = "语音内容不足，建议多说一些"
        else:
            quality = "良好"

        return HealthReport(
            overall_risk_level=overall,
            overall_score=round(overall_score, 1),
            diseases=diseases,
            feature_summary=feature_summary,
            recommendations=recommendations,
            recording_quality=quality,
        )


__all__ = ["DiseaseDetector", "DiseaseRisk", "HealthReport", "DISEASE_REGISTRY"]
