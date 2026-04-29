"""
VoiceHealth — 疾病检测引擎

基于声学特征的多疾病筛查系统。
覆盖5大类别25种疾病/健康状态。

架构设计：
- 每种疾病类别有独立的检测模型
- 支持Mock模式（无训练数据时的降级）
- 输出风险评分 0-100 + 置信度
- 所有25种疾病均有详细的基于规则的检测逻辑
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

        # ═══════════════════════════════════════════════════
        # 心理健康类疾病
        # ═══════════════════════════════════════════════════

        if disease_id == "depression":
            # 医学依据：抑郁症患者语音特征包括语速降低(prosodic slowing)、
            # 基频变异减少(reduced F0 variability)、停顿增多(increased pause duration)
            if feat.speech_rate < 2.5:
                score += 20; markers_found.append("语速偏低")
            if feat.f0_std < 20:
                score += 18; markers_found.append("音调单调")
            if feat.pause_ratio > 0.30:
                score += 18; markers_found.append("停顿较多")
            if feat.rms_mean < 0.02:
                score += 12; markers_found.append("能量偏低")
            if feat.energy_std < 0.001:
                score += 10; markers_found.append("动态范围小")
            if feat.f0_range < 80:
                score += 12; markers_found.append("音域狭窄")

        elif disease_id == "anxiety":
            # 医学依据：焦虑导致交感神经激活，表现为语速加快(tachylalia)、
            # 基频升高(elevated F0)、声音微颤(increased jitter/shimmer)
            if feat.speech_rate > 4.5:
                score += 22; markers_found.append("语速偏快")
            if feat.f0_mean > 200:
                score += 15; markers_found.append("音调偏高")
            if feat.jitter_local > 0.02:
                score += 18; markers_found.append("Jitter增大")
            if feat.shimmer_local > 0.05:
                score += 12; markers_found.append("Shimmer增大")
            if feat.f0_std > 60:
                score += 10; markers_found.append("音调波动大")
            if feat.rms_std > 0.03:
                score += 10; markers_found.append("能量波动大")
            if feat.zero_crossing_rate_mean > 0.12:
                score += 8; markers_found.append("噪声增多")

        elif disease_id == "burnout":
            # 医学依据：职业倦怠导致发声动力不足、语速降低、
            # 基频下降(reduced pitch)、能量衰减(decreased energy)
            if feat.speech_rate < 2.0:
                score += 22; markers_found.append("语速缓慢")
            if feat.rms_mean < 0.015:
                score += 20; markers_found.append("能量极低")
            if feat.pause_ratio > 0.35:
                score += 18; markers_found.append("停顿异常")
            if feat.f0_range < 60:
                score += 15; markers_found.append("音域狭窄")
            if feat.f0_std < 18:
                score += 12; markers_found.append("音调单调")
            if feat.energy_std < 0.0008:
                score += 10; markers_found.append("缺乏活力")

        elif disease_id == "stress":
            # 医学依据：心理压力引起喉部肌肉紧张，导致基频升高、
            # Jitter/Shimmer增大、HNR下降(degraded voice quality)
            if feat.f0_mean > 180:
                score += 18; markers_found.append("音调升高")
            if feat.jitter_local > 0.015:
                score += 15; markers_found.append("Jitter增大")
            if feat.shimmer_local > 0.04:
                score += 15; markers_found.append("Shimmer增大")
            if feat.hnr_mean < 12:
                score += 15; markers_found.append("HNR下降")
            if feat.rms_std > 0.025:
                score += 10; markers_found.append("能量波动大")
            if feat.spectral_centroid_std > 400:
                score += 10; markers_found.append("频谱不稳定")

        elif disease_id == "social_isolation":
            # 医学依据：长期社交孤立导致语音韵律单调化(prosodic flattening)、
            # 语速减慢、基频变异降低、停顿模式改变
            if feat.speech_rate < 2.2:
                score += 20; markers_found.append("语速减慢")
            if feat.f0_std < 15:
                score += 18; markers_found.append("韵律单调")
            if feat.pause_ratio > 0.35:
                score += 15; markers_found.append("停顿较多")
            if feat.f0_range < 70:
                score += 15; markers_found.append("音调单一")
            if feat.rms_mean < 0.018:
                score += 12; markers_found.append("能量偏低")
            if feat.pause_count > 8:
                score += 10; markers_found.append("响应延迟迹象")

        # ═══════════════════════════════════════════════════
        # 认知退行类疾病
        # ═══════════════════════════════════════════════════

        elif disease_id == "parkinsons":
            # 医学依据：帕金森病影响发声肌群，Jitter/Shimmer显著升高、
            # HNR下降(impaired phonation)、语速不稳(fluctuating speech rate)
            if feat.jitter_local > 0.025:
                score += 28; markers_found.append("Jitter显著↑")
            if feat.shimmer_local > 0.06:
                score += 22; markers_found.append("Shimmer↑")
            if feat.hnr_mean < 10:
                score += 20; markers_found.append("HNR↓")
            if feat.speech_rate < 2.0 or feat.speech_rate > 5.0:
                score += 15; markers_found.append("语速不稳")
            if feat.f0_std > 50:
                score += 10; markers_found.append("发声震颤迹象")
            if feat.rms_mean < 0.015:
                score += 8; markers_found.append("发声力量不足")

        elif disease_id == "alzheimers":
            # 医学依据：AD患者出现词汇检索困难(word-finding difficulty)、
            # 句法简化(syntactic simplification)、停顿模式异常(abnormal pausing)
            # 通过停顿比率、语速、能量模式推断
            if feat.pause_ratio > 0.40:
                score += 22; markers_found.append("停顿模式异常")
            if feat.speech_rate < 2.0:
                score += 18; markers_found.append("语速偏低")
            if feat.f0_std < 18:
                score += 15; markers_found.append("音调单调(词汇检索延迟)")
            if feat.pause_count > 10:
                score += 15; markers_found.append("频繁停顿(检索困难)")
            if feat.rms_std < 0.0008:
                score += 12; markers_found.append("动态范围缩小")
            if feat.energy_std < 0.0006:
                score += 10; markers_found.append("发声动力不足")

        elif disease_id == "mci":
            # 医学依据：MCI患者F0变异增大(increased F0 variability)、
            # 停顿时间增加、语速略有下降
            if feat.f0_std > 55:
                score += 18; markers_found.append("F0变异增大")
            if feat.pause_ratio > 0.32:
                score += 18; markers_found.append("停顿增加")
            if feat.speech_rate < 2.3:
                score += 15; markers_found.append("语速下降")
            if feat.f0_range > 200:
                score += 12; markers_found.append("音域波动异常")
            if feat.pause_count > 7:
                score += 12; markers_found.append("停顿频次高")
            if feat.hnr_mean < 13:
                score += 10; markers_found.append("发声质量下降")

        elif disease_id == "frailty":
            # 医学依据：老年衰弱综合征导致发声肌肉力量下降、
            # 音量减低(reduced loudness)、语速极慢、音调偏低
            if feat.rms_mean < 0.01:
                score += 28; markers_found.append("音量极低")
            if feat.speech_rate < 1.5:
                score += 22; markers_found.append("语速极慢")
            if feat.f0_mean < 80:
                score += 18; markers_found.append("发声无力")
            if feat.pause_ratio > 0.5:
                score += 15; markers_found.append("停顿极多")
            if feat.f0_std < 12:
                score += 10; markers_found.append("音调变化极小")
            if feat.voiced_ratio < 0.3:
                score += 8; markers_found.append("浊音比例极低")

        # ═══════════════════════════════════════════════════
        # 呼吸系统疾病
        # ═══════════════════════════════════════════════════

        elif disease_id == "copd":
            # 医学依据：COPD导致呼气气流受限(expiratory airflow limitation)，
            # 表现为停顿增多、浊音比例下降、频谱平坦化
            if feat.spectral_flatness_mean > 0.10:
                score += 18; markers_found.append("频谱平坦")
            if feat.pause_ratio > 0.40:
                score += 22; markers_found.append("呼气缩短")
            if feat.rms_std > 0.03:
                score += 15; markers_found.append("气流不稳")
            if feat.speech_rate < 2.0:
                score += 15; markers_found.append("语速缓慢")
            if feat.voiced_ratio < 0.4:
                score += 15; markers_found.append("浊音比例↓")
            if feat.pause_count > 10:
                score += 10; markers_found.append("频繁换气")

        elif disease_id == "asthma":
            # 医学依据：哮喘导致气道阻塞，表现为呼吸音异常、
            # 语音中断、气流不稳定、停顿比升高
            if feat.pause_ratio > 0.35:
                score += 20; markers_found.append("停顿增多(呼吸困难)")
            if feat.rms_std > 0.025:
                score += 18; markers_found.append("气流不稳")
            if feat.voiced_ratio < 0.45:
                score += 15; markers_found.append("浊音比例下降")
            if feat.spectral_flatness_mean > 0.08:
                score += 15; markers_found.append("呼吸音混入")
            if feat.speech_rate < 2.5:
                score += 12; markers_found.append("语速受限")
            if feat.energy_std > 0.02:
                score += 10; markers_found.append("能量波动(间断)")

        elif disease_id == "respiratory_distress":
            # 医学依据：呼吸窘迫时说话短促(breathiness)、
            # 呼吸频率加快、发声困难、停顿急剧增多
            if feat.pause_ratio > 0.45:
                score += 25; markers_found.append("停顿极多")
            if feat.speech_rate < 1.8:
                score += 20; markers_found.append("语速极慢")
            if feat.voiced_ratio < 0.35:
                score += 18; markers_found.append("发声困难")
            if feat.rms_mean < 0.012:
                score += 15; markers_found.append("声音微弱")
            if feat.pause_count > 12:
                score += 12; markers_found.append("呼吸频率↑")
            if feat.hnr_mean < 8:
                score += 10; markers_found.append("发声质量差")

        # ═══════════════════════════════════════════════════
        # 心血管疾病
        # ═══════════════════════════════════════════════════

        elif disease_id == "hypertension":
            # 医学依据：高血压影响自主神经系统，导致语速变化、
            # 音调紧张度升高(vocal tension)、呼吸模式改变
            if feat.speech_rate > 4.2 or feat.speech_rate < 2.2:
                score += 18; markers_found.append("语速异常")
            if feat.f0_mean > 185:
                score += 18; markers_found.append("音调紧张")
            if feat.jitter_local > 0.018:
                score += 15; markers_found.append("微颤增加")
            if feat.shimmer_local > 0.045:
                score += 12; markers_found.append("振幅不稳定")
            if feat.pause_ratio > 0.30:
                score += 12; markers_found.append("呼吸模式改变")
            if feat.hnr_mean < 11:
                score += 10; markers_found.append("发声紧张度↑")

        elif disease_id == "heart_failure":
            # 匱学依据：心衰导致全身供血不足，表现为呼吸困难(dyspnea)、
            # 语速极慢、音量微弱、频繁停顿以换气
            if feat.pause_ratio > 0.42:
                score += 22; markers_found.append("呼吸困难")
            if feat.speech_rate < 1.8:
                score += 20; markers_found.append("语速极慢")
            if feat.rms_mean < 0.012:
                score += 18; markers_found.append("音量微弱")
            if feat.pause_count > 10:
                score += 15; markers_found.append("频繁停顿")
            if feat.voiced_ratio < 0.4:
                score += 12; markers_found.append("发声受限")
            if feat.f0_std < 15:
                score += 8; markers_found.append("发声无力")

        # ═══════════════════════════════════════════════════
        # 代谢疾病
        # ═══════════════════════════════════════════════════

        elif disease_id == "diabetes_t2":
            # 医学依据：高血糖影响神经和肌肉功能(neuropathy)，
            # 导致舌运动变化、发音清晰度下降、共振峰偏移
            if abs(feat.formant_f1 - 500) > 200:
                score += 15; markers_found.append("F1偏移")
            if feat.spectral_flatness_mean > 0.08:
                score += 15; markers_found.append("清晰度下降")
            if feat.speech_rate < 2.5:
                score += 10; markers_found.append("语速偏慢")
            if abs(feat.formant_f2 - 1500) > 300:
                score += 12; markers_found.append("F2偏移(舌位异常)")
            if feat.jitter_local > 0.018:
                score += 10; markers_found.append("发声微颤")
            if feat.hnr_mean < 12:
                score += 10; markers_found.append("发声质量下降")

        elif disease_id == "thyroid":
            # 医学依据：甲状腺功能异常(甲亢/甲减)影响声带水肿和
            # 喉部肌肉张力，导致音调变化、语速异常
            if feat.f0_mean > 210 or feat.f0_mean < 90:
                score += 20; markers_found.append("音调异常")
            if feat.speech_rate > 4.5 or feat.speech_rate < 2.0:
                score += 18; markers_found.append("语速异常")
            if feat.shimmer_local > 0.06:
                score += 15; markers_found.append("声带水肿特征")
            if feat.f0_std > 50 or feat.f0_std < 15:
                score += 15; markers_found.append("音调变化异常")
            if feat.jitter_local > 0.02:
                score += 12; markers_found.append("发声不稳定")
            if feat.hnr_mean < 10:
                score += 10; markers_found.append("声音质量下降")

        # ═══════════════════════════════════════════════════
        # 健康状态评估
        # ═══════════════════════════════════════════════════

        elif disease_id == "sleep_quality":
            # 医学依据：睡眠不足导致语音活力下降(reduced vocal vigor)、
            # 语速减慢、音调单调、能量水平低
            if feat.rms_mean < 0.02:
                score += 18; markers_found.append("能量不足")
            if feat.speech_rate < 3.0:
                score += 15; markers_found.append("语速偏慢")
            if feat.f0_std < 25:
                score += 15; markers_found.append("音调单调")
            if feat.pause_count > 5:
                score += 10; markers_found.append("频繁停顿")
            if feat.energy_std < 0.001:
                score += 12; markers_found.append("缺乏活力")
            if feat.f0_range < 80:
                score += 10; markers_found.append("音域缩小")

        elif disease_id == "fatigue":
            # 医学依据：疲劳导致发声动力不足(reduced subglottic pressure)、
            # RMS能量下降、语速减慢、停顿增多、基频下降
            if feat.rms_mean < 0.015:
                score += 22; markers_found.append("RMS偏低")
            if feat.speech_rate < 2.5:
                score += 18; markers_found.append("语速缓慢")
            if feat.pause_ratio > 0.35:
                score += 18; markers_found.append("停顿较多")
            if feat.f0_mean < 100:
                score += 15; markers_found.append("音调偏低")
            if feat.energy_std < 0.001:
                score += 10; markers_found.append("能量波动小")
            if feat.voiced_ratio < 0.45:
                score += 10; markers_found.append("浊音比例下降")

        elif disease_id == "alcohol_influence":
            # 医学依据：酒精影响小脑和运动皮层，导致发音协调性下降、
            # Jitter/Shimmer显著升高、频谱不稳定、发音模糊
            if feat.jitter_local > 0.03:
                score += 28; markers_found.append("Jitter↑↑")
            if feat.shimmer_local > 0.08:
                score += 22; markers_found.append("Shimmer↑↑")
            if feat.spectral_centroid_std > 500:
                score += 18; markers_found.append("发音不稳定")
            if feat.zero_crossing_rate_mean > 0.15:
                score += 12; markers_found.append("噪声增多")
            if feat.hnr_mean < 8:
                score += 10; markers_found.append("声音质量严重下降")
            if feat.f0_std > 60:
                score += 8; markers_found.append("音调失控")

        elif disease_id == "pain_level":
            # 医学依据：疼痛导致喉部肌肉紧张(vocal tension)、
            # 音调升高、能量降低、呼吸抑制、发声受限
            if feat.f0_mean > 190:
                score += 20; markers_found.append("音调紧张")
            if feat.rms_mean < 0.015:
                score += 18; markers_found.append("能量降低")
            if feat.jitter_local > 0.02:
                score += 15; markers_found.append("发声颤抖")
            if feat.pause_ratio > 0.35:
                score += 15; markers_found.append("呼吸抑制")
            if feat.shimmer_local > 0.05:
                score += 12; markers_found.append("振幅不稳定")
            if feat.hnr_mean < 10:
                score += 10; markers_found.append("发声受限")

        elif disease_id == "emotional_state":
            # 医学依据：情绪状态通过多维度语音指标综合评估，
            # 包括F0变异、能量动态、语速模式、频谱特征
            if feat.f0_std > 55:
                score += 15; markers_found.append("F0变异增大(情绪激动)")
            if feat.f0_std < 15:
                score += 12; markers_found.append("F0变异减小(情绪低落)")
            if feat.energy_std > 0.025:
                score += 12; markers_found.append("能量动态大")
            if feat.speech_rate > 4.5:
                score += 12; markers_found.append("语速偏快(焦虑/激动)")
            if feat.speech_rate < 2.0:
                score += 12; markers_found.append("语速偏慢(低落)")
            if feat.spectral_centroid_std > 400:
                score += 10; markers_found.append("频谱倾斜变化")

        elif disease_id == "voice_disorder":
            # 医学依据：声带病理(如声带结节、息肉)导致Jitter/Shimmer
            # 显著升高、HNR严重下降、发声困难
            if feat.jitter_local > 0.04:
                score += 28; markers_found.append("Jitter严重↑")
            if feat.shimmer_local > 0.10:
                score += 25; markers_found.append("Shimmer严重↑")
            if feat.hnr_mean < 5:
                score += 22; markers_found.append("HNR严重↓")
            if feat.rms_mean < 0.01:
                score += 12; markers_found.append("发声困难")
            if feat.voiced_ratio < 0.35:
                score += 8; markers_found.append("浊音比例极低")

        elif disease_id == "hearing_impairment":
            # 医学依据：听力下降导致自我语音监控失效(self-monitoring deficit)，
            # 表现为音量过大、音调控制偏差、共振峰偏移
            if feat.rms_mean > 0.06:
                score += 22; markers_found.append("音量异常大")
            if abs(feat.formant_f1 - 500) > 250:
                score += 18; markers_found.append("F1偏移(音调控制偏差)")
            if abs(feat.formant_f2 - 1500) > 400:
                score += 15; markers_found.append("F2偏移(共振峰异常)")
            if feat.f0_std > 60:
                score += 12; markers_found.append("音调波动异常")
            if feat.shimmer_local > 0.06:
                score += 10; markers_found.append("振幅控制不佳")
            if feat.spectral_centroid_mean > 3000:
                score += 10; markers_found.append("频谱重心偏高")

        elif disease_id == "cognitive_load":
            # 医学依据：高认知负荷导致语速下降(speech rate reduction)、
            # 停顿增加(increased pausing)、发音不流畅(disfluency)
            if feat.speech_rate < 2.2:
                score += 20; markers_found.append("语速下降")
            if feat.pause_ratio > 0.35:
                score += 18; markers_found.append("停顿增加")
            if feat.pause_count > 8:
                score += 15; markers_found.append("频繁停顿")
            if feat.f0_std > 50:
                score += 12; markers_found.append("音调不稳(负荷反应)")
            if feat.rms_std > 0.025:
                score += 10; markers_found.append("能量波动(不流畅)")
            if feat.voiced_ratio < 0.45:
                score += 10; markers_found.append("浊音比例下降")

        elif disease_id == "autism_spectrum":
            # 医学依据：ASD相关语音韵律异常(prosodic abnormalities)，
            # 表现为音调单一(monotone)、节奏刻板(stereotyped rhythm)、
            # 韵律规律性异常
            if feat.f0_std < 15:
                score += 22; markers_found.append("音调单一")
            if feat.f0_range < 60:
                score += 18; markers_found.append("音域极窄")
            if feat.energy_std < 0.0008:
                score += 15; markers_found.append("能量变化极小(刻板)")
            if feat.spectral_centroid_std < 200:
                score += 12; markers_found.append("频谱变化小(规律性强)")
            if feat.rms_std < 0.005:
                score += 12; markers_found.append("节奏刻板")
            if feat.tempo and feat.tempo > 0:
                # 节奏规律性：如果tempo标准差小，表示节奏刻板
                pass  # tempo为单一值，用其他指标代替
            if feat.speech_rate > 3.5 and feat.f0_std < 20:
                score += 10; markers_found.append("语速快但音调单调")

        # ── 兜底：不应到达此处 ──
        else:
            # 通用启发式：基于特征值异常
            vec = feat.to_vector()
            norm = np.linalg.norm(vec[:10])
            if norm > 2.0:
                score += min(30, norm * 5)
                markers_found.append("声学特征偏移")

        # ── 最终处理 ──
        # 添加小幅度随机扰动模拟模型不确定性（降低为std=1）
        score = np.clip(score + np.random.normal(0, 1), 0, 100)

        # 调整置信度计算：基于匹配条件数量和分数综合评估
        n_markers = len(markers_found)
        marker_confidence = min(0.45, n_markers * 0.08)  # 每个标记贡献0.08，上限0.45
        score_confidence = min(0.40, score / 250)         # 分数贡献，上限0.40
        confidence = min(0.88, 0.30 + marker_confidence + score_confidence)

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
