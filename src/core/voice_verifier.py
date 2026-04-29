"""
VoiceHealth — 语音验证引擎

1. 活体检测（Liveness Detection）
   - 检测是否为真人发声
   - 防止播放录音、合成语音
   
2. 朗读验证（Reading Verification）
   - 语音识别（ASR）
   - 与预设文本比对
   - 检查朗读完整度
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


@dataclass
class LivenessResult:
    """活体检测结果"""
    is_live: bool = False
    confidence: float = 0.0
    score: float = 0.0
    checks: Dict[str, bool] = None
    details: Dict[str, float] = None
    reason: str = ""
    
    def __post_init__(self):
        if self.checks is None:
            self.checks = {}
        if self.details is None:
            self.details = {}


@dataclass
class ReadingVerification:
    """朗读验证结果"""
    is_valid: bool = False
    confidence: float = 0.0
    recognized_text: str = ""
    expected_text: str = ""
    match_ratio: float = 0.0
    word_count: int = 0
    expected_word_count: int = 0
    missing_words: List[str] = None
    extra_words: List[str] = None
    reason: str = ""
    
    def __post_init__(self):
        if self.missing_words is None:
            self.missing_words = []
        if self.extra_words is None:
            self.extra_words = []


# 预设朗读文本库
READING_TEXTS = [
    {
        "id": "standard_1",
        "text": "春天来了，花儿开了，小鸟在枝头唱歌。阳光温暖地照在大地上，万物复苏，生机勃勃。我喜欢在这样的日子里，和朋友们一起去公园散步，感受大自然的美好。",
        "keywords": ["春天", "花儿", "小鸟", "阳光", "公园", "散步", "大自然"]
    },
    {
        "id": "standard_2", 
        "text": "健康是人生最宝贵的财富。我们应该养成良好的生活习惯，早睡早起，坚持锻炼，合理饮食。每天保持愉快的心情，积极面对生活中的挑战，这样才能拥有健康的身体。",
        "keywords": ["健康", "习惯", "锻炼", "饮食", "心情", "挑战", "身体"]
    },
    {
        "id": "standard_3",
        "text": "科技改变了我们的生活。智能手机让我们随时随地与朋友联系，互联网让我们获取信息更加便捷。人工智能技术正在快速发展，未来将会有更多创新应用出现，让生活变得更加美好。",
        "keywords": ["科技", "手机", "互联网", "人工智能", "创新", "应用", "美好"]
    },
    {
        "id": "standard_4",
        "text": "学习是一辈子的事情。无论年龄大小，我们都应该保持好奇心，不断探索新知识。读书是获取知识的重要途径，通过阅读，我们可以了解不同的文化，开阔视野，丰富内心世界。",
        "keywords": ["学习", "好奇心", "知识", "读书", "文化", "视野", "内心"]
    },
    {
        "id": "standard_5",
        "text": "家庭是我们温暖的港湾。父母的关爱，兄弟姐妹的陪伴，让我们感受到亲情的力量。无论遇到什么困难，家人总是我们最坚强的后盾。珍惜与家人在一起的每一刻，创造美好的回忆。",
        "keywords": ["家庭", "父母", "关爱", "亲情", "困难", "家人", "回忆"]
    }
]


class LivenessDetector:
    """活体检测器"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        
    def detect(self, audio_path: str) -> LivenessResult:
        """
        检测音频是否为真人发声
        
        检测维度：
        1. 背景噪声分析（录音通常有环境噪声）
        2. 频谱连续性（录音会有压缩伪影）
        3. 动态范围（录音会压缩动态）
        4. 微表情分析（真人发声有微小变化）
        """
        if not HAS_LIBROSA:
            return self._fallback_detect(audio_path)
            
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            result = LivenessResult()
            
            # 检测1：背景噪声分析
            noise_check, noise_score = self._check_background_noise(y, sr)
            result.checks['background_noise'] = noise_check
            result.details['noise_score'] = noise_score
            
            # 检测2：频谱连续性
            spectral_check, spectral_score = self._check_spectral_continuity(y, sr)
            result.checks['spectral_continuity'] = spectral_check
            result.details['spectral_score'] = spectral_score
            
            # 检测3：动态范围
            dynamic_check, dynamic_score = self._check_dynamic_range(y)
            result.checks['dynamic_range'] = dynamic_check
            result.details['dynamic_score'] = dynamic_score
            
            # 检测4：微变化分析
            micro_check, micro_score = self._check_micro_variations(y, sr)
            result.checks['micro_variations'] = micro_check
            result.details['micro_score'] = micro_score
            
            # 检测5：谐波结构
            harmonic_check, harmonic_score = self._check_harmonic_structure(y, sr)
            result.checks['harmonic_structure'] = harmonic_check
            result.details['harmonic_score'] = harmonic_score
            
            # 综合评分
            checks_passed = sum(result.checks.values())
            total_checks = len(result.checks)
            
            result.score = (
                noise_score * 0.2 +
                spectral_score * 0.2 +
                dynamic_score * 0.2 +
                micro_score * 0.25 +
                harmonic_score * 0.15
            )
            
            result.confidence = result.score
            result.is_live = checks_passed >= 3 and result.score >= 0.6
            
            if not result.is_live:
                failed = [k for k, v in result.checks.items() if not v]
                result.reason = f"可能不是真人发声，未通过检测：{', '.join(failed)}"
            
            return result
            
        except Exception as e:
            return LivenessResult(
                is_live=False,
                reason=f"检测失败：{str(e)}"
            )
    
    def _check_background_noise(self, y: np.ndarray, sr: int) -> Tuple[bool, float]:
        """检查背景噪声模式"""
        # 真人录音通常有自然的环境噪声
        # 播放录音会有更多高频噪声或特定噪声模式
        
        # 计算噪声底噪
        rms = librosa.feature.rms(y=y)[0]
        noise_floor = np.percentile(rms, 10)
        
        # 检查噪声频谱
        stft = np.abs(librosa.stft(y))
        noise_spectrum = np.mean(stft[:, :10], axis=1)  # 前10帧作为噪声估计
        
        # 真人录音的噪声通常更自然
        noise_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # 评分：噪声平坦度在合理范围内
        if 0.01 < noise_flatness < 0.3:
            score = 0.8
        elif 0.005 < noise_flatness < 0.5:
            score = 0.6
        else:
            score = 0.3
            
        return score > 0.5, score
    
    def _check_spectral_continuity(self, y: np.ndarray, sr: int) -> Tuple[bool, float]:
        """检查频谱连续性"""
        # 真人发声的频谱变化更自然
        # 录音播放会有压缩伪影
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 计算MFCC的连续性
        mfcc_diff = np.diff(mfccs, axis=1)
        continuity = np.mean(np.abs(mfcc_diff))
        
        # 真人发声的连续性变化更平滑
        if continuity < 5:
            score = 0.8
        elif continuity < 10:
            score = 0.6
        else:
            score = 0.4
            
        return score > 0.5, score
    
    def _check_dynamic_range(self, y: np.ndarray) -> Tuple[bool, float]:
        """检查动态范围"""
        # 真人发声有更自然的动态变化
        # 录音播放通常动态范围被压缩
        
        rms = librosa.feature.rms(y=y)[0]
        dynamic_range = np.max(rms) - np.min(rms)
        dynamic_ratio = np.max(rms) / (np.min(rms) + 1e-8)
        
        # 真人发声的动态范围通常在合理范围内
        if 0.1 < dynamic_range < 0.8 and 5 < dynamic_ratio < 50:
            score = 0.8
        elif 0.05 < dynamic_range < 0.9 and 3 < dynamic_ratio < 100:
            score = 0.6
        else:
            score = 0.4
            
        return score > 0.5, score
    
    def _check_micro_variations(self, y: np.ndarray, sr: int) -> Tuple[bool, float]:
        """检查微变化"""
        # 真人发声有微小的抖动和变化
        # 录音播放相对更"完美"
        
        # 计算基频的微变化
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr
        )
        
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 10:
            # 计算基频的变异系数
            f0_cv = np.std(f0_valid) / (np.mean(f0_valid) + 1e-8)
            
            # 真人发声的变异系数通常在合理范围内
            if 0.01 < f0_cv < 0.15:
                score = 0.9
            elif 0.005 < f0_cv < 0.2:
                score = 0.7
            else:
                score = 0.4
        else:
            score = 0.5
            
        return score > 0.5, score
    
    def _check_harmonic_structure(self, y: np.ndarray, sr: int) -> Tuple[bool, float]:
        """检查谐波结构"""
        # 真人发声有自然的谐波结构
        # 录音播放可能有压缩或失真
        
        harmonic, percussive = librosa.effects.hpss(y)
        
        # 计算谐波与打击乐的比例
        h_energy = np.sum(harmonic ** 2)
        p_energy = np.sum(percussive ** 2)
        
        if p_energy > 0:
            hp_ratio = h_energy / p_energy
        else:
            hp_ratio = 10
            
        # 真人发声的谐波比例通常较高
        if hp_ratio > 2:
            score = 0.8
        elif hp_ratio > 1:
            score = 0.6
        else:
            score = 0.4
            
        return score > 0.5, score
    
    def _fallback_detect(self, audio_path: str) -> LivenessResult:
        """降级检测（无librosa时）"""
        return LivenessResult(
            is_live=True,
            confidence=0.5,
            score=0.5,
            reason="降级模式：无法进行完整检测"
        )


class ReadingVerifier:
    """朗读验证器"""
    
    def __init__(self):
        self.reading_texts = {t["id"]: t for t in READING_TEXTS}
    
    def get_random_text(self) -> Dict:
        """获取随机朗读文本"""
        import random
        return random.choice(READING_TEXTS)
    
    def verify(self, recognized_text: str, expected_text_id: str) -> ReadingVerification:
        """
        验证朗读内容
        
        Args:
            recognized_text: 语音识别出的文字
            expected_text_id: 预期文本ID
            
        Returns:
            ReadingVerification: 验证结果
        """
        result = ReadingVerification()
        
        if expected_text_id not in self.reading_texts:
            result.reason = "未知的朗读文本ID"
            return result
        
        expected = self.reading_texts[expected_text_id]
        result.expected_text = expected["text"]
        result.recognized_text = recognized_text
        
        # 清理文本
        clean_recognized = self._clean_text(recognized_text)
        clean_expected = self._clean_text(expected["text"])
        
        # 分词
        recognized_words = self._tokenize(clean_recognized)
        expected_words = self._tokenize(clean_expected)
        
        result.word_count = len(recognized_words)
        result.expected_word_count = len(expected_words)
        
        # 计算匹配度
        match_ratio = self._calculate_match_ratio(recognized_words, expected_words)
        result.match_ratio = match_ratio
        
        # 检查关键词
        keyword_match = self._check_keywords(clean_recognized, expected["keywords"])
        
        # 计算置信度
        result.confidence = (match_ratio * 0.6 + keyword_match * 0.4)
        
        # 判断是否有效
        result.is_valid = result.confidence >= 0.5 and match_ratio >= 0.4
        
        if not result.is_valid:
            if match_ratio < 0.4:
                result.reason = f"朗读内容匹配度过低：{match_ratio:.1%}"
            else:
                result.reason = f"朗读验证未通过：{result.confidence:.1%}"
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 去除标点符号和空格
        text = re.sub(r'[，。！？、；：""''（）《》【】\s]+', '', text)
        text = re.sub(r'[,.!?;:\'"()\[\]{}<>\s]+', '', text)
        return text.lower()
    
    def _tokenize(self, text: str) -> List[str]:
        """中文分词（简单按字分）"""
        return list(text)
    
    def _calculate_match_ratio(self, recognized: List[str], expected: List[str]) -> float:
        """计算匹配度"""
        if not expected:
            return 0.0
        
        # 使用最长公共子序列计算相似度
        lcs_length = self._lcs_length(recognized, expected)
        return lcs_length / len(expected)
    
    def _lcs_length(self, a: List[str], b: List[str]) -> int:
        """最长公共子序列长度"""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _check_keywords(self, text: str, keywords: List[str]) -> float:
        """检查关键词匹配"""
        if not keywords:
            return 1.0
        
        matched = sum(1 for kw in keywords if kw in text)
        return matched / len(keywords)


# 语音识别（ASR）集成
class SpeechRecognizer:
    """语音识别器"""
    
    def __init__(self):
        # 这里可以集成各种ASR服务
        # 1. 微信同声传译插件
        # 2. 百度语音识别
        # 3. 阿里云语音识别
        # 4. 讯飞语音识别
        pass
    
    def recognize(self, audio_path: str, language: str = "zh-CN") -> str:
        """
        识别语音内容
        
        注意：实际项目中需要集成真实的ASR服务
        这里返回模拟结果用于开发测试
        """
        # TODO: 集成真实ASR服务
        # 方案1: 使用微信同声传译插件（免费）
        # 方案2: 使用百度语音识别API
        # 方案3: 使用阿里云智能语音交互
        
        return ""  # 返回空字符串，需要集成真实ASR


# 导出
__all__ = [
    "LivenessDetector", 
    "LivenessResult",
    "ReadingVerifier",
    "ReadingVerification", 
    "SpeechRecognizer",
    "READING_TEXTS"
]
