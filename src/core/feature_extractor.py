"""
VoiceHealth — 声学特征提取引擎 v2.0

完整的语音生物标志物提取系统。
基于librosa + numpy的双引擎架构。

59维声学特征向量：
- MFCC (26维): 均值+标准差 × 13系数
- 基频F0 (6维): 均值/标准差/最小/最大/范围/中位数
- Jitter/Shimmer (6维): 声带振动稳定性
- HNR (2维): 谐波噪声比
- 频谱特征 (6维): 质心/带宽/平坦度/滚降
- 韵律特征 (9维): 语速/停顿/时长/节奏
- 共振峰 (4维): F1-F4
- 能量特征 (2维): RMS均值/标准差
"""

import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


@dataclass
class AcousticFeatures:
    """声学特征数据结构"""
    # 基频
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_min: float = 0.0
    f0_max: float = 0.0
    f0_range: float = 0.0
    f0_median: float = 0.0

    # Jitter（频率微扰）
    jitter_local: float = 0.0
    jitter_rap: float = 0.0
    jitter_ppq5: float = 0.0

    # Shimmer（振幅微扰）
    shimmer_local: float = 0.0
    shimmer_apq5: float = 0.0
    shimmer_apq11: float = 0.0

    # HNR（谐波噪声比）
    hnr_mean: float = 0.0
    hnr_std: float = 0.0

    # MFCC
    mfcc_means: List[float] = field(default_factory=lambda: [0.0] * 13)
    mfcc_stds: List[float] = field(default_factory=lambda: [0.0] * 13)

    # 频谱特征
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_flatness_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0
    zero_crossing_rate_mean: float = 0.0

    # 韵律特征
    speech_rate: float = 0.0
    pause_ratio: float = 0.0
    pause_count: int = 0
    total_duration: float = 0.0
    voiced_ratio: float = 0.0
    energy_std: float = 0.0
    rms_mean: float = 0.0
    rms_std: float = 0.0
    tempo: float = 0.0

    # 共振峰
    formant_f1: float = 0.0
    formant_f2: float = 0.0
    formant_f3: float = 0.0
    formant_f4: float = 0.0

    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            self.f0_mean, self.f0_std, self.f0_min, self.f0_max,
            self.f0_range, self.f0_median,
            self.jitter_local, self.jitter_rap, self.jitter_ppq5,
            self.shimmer_local, self.shimmer_apq5, self.shimmer_apq11,
            self.hnr_mean, self.hnr_std,
            *self.mfcc_means, *self.mfcc_stds,
            self.spectral_centroid_mean, self.spectral_centroid_std,
            self.spectral_bandwidth_mean, self.spectral_flatness_mean,
            self.spectral_rolloff_mean, self.zero_crossing_rate_mean,
            self.speech_rate, self.pause_ratio, self.pause_count,
            self.total_duration, self.voiced_ratio, self.energy_std,
            self.rms_mean, self.rms_std, self.tempo,
            self.formant_f1, self.formant_f2, self.formant_f3, self.formant_f4,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class FeatureExtractor:
    """声学特征提取引擎"""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def extract(self, audio_path: str) -> AcousticFeatures:
        """从音频文件提取完整声学特征"""
        if HAS_LIBROSA:
            return self._extract_with_librosa(audio_path)
        else:
            return self._extract_with_numpy(audio_path)

    def _extract_with_librosa(self, path: str) -> AcousticFeatures:
        """使用librosa提取特征"""
        y, sr = librosa.load(path, sr=self.sr)

        features = AcousticFeatures()
        features.total_duration = len(y) / sr

        # 基频提取
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 0:
            features.f0_mean = float(np.mean(f0_valid))
            features.f0_std = float(np.std(f0_valid))
            features.f0_min = float(np.min(f0_valid))
            features.f0_max = float(np.max(f0_valid))
            features.f0_range = float(features.f0_max - features.f0_min)
            features.f0_median = float(np.median(f0_valid))

        # Jitter近似
        if len(f0_valid) > 2:
            f0_diff = np.abs(np.diff(f0_valid))
            features.jitter_local = float(np.mean(f0_diff) / np.mean(f0_valid))
            features.jitter_rap = float(np.median(f0_diff) / np.mean(f0_valid))
            if len(f0_valid) > 4:
                ppq5_diff = []
                for i in range(2, len(f0_valid) - 2):
                    avg_neighbors = np.mean(f0_valid[i-2:i+3])
                    ppq5_diff.append(abs(f0_valid[i] - avg_neighbors))
                features.jitter_ppq5 = float(np.mean(ppq5_diff) / np.mean(f0_valid))

        # Shimmer近似
        rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        if len(rms_frames) > 2:
            rms_diff = np.abs(np.diff(rms_frames))
            features.shimmer_local = float(np.mean(rms_diff) / (np.mean(rms_frames) + 1e-8))
            features.shimmer_apq5 = float(np.median(rms_diff) / (np.mean(rms_frames) + 1e-8))
            if len(rms_frames) > 10:
                apq11_diff = []
                for i in range(5, len(rms_frames) - 5):
                    avg_n = np.mean(rms_frames[i-5:i+6])
                    apq11_diff.append(abs(rms_frames[i] - avg_n))
                features.shimmer_apq11 = float(np.mean(apq11_diff) / (np.mean(rms_frames) + 1e-8))

        # HNR近似
        harmonic, percussive = librosa.effects.hpss(y)
        h_power = np.mean(harmonic ** 2)
        n_power = np.mean(percussive ** 2)
        if n_power > 0:
            features.hnr_mean = float(10 * np.log10(h_power / (n_power + 1e-8)))
        hnr_frames = []
        for i in range(0, len(y) - 2048, 512):
            seg = y[i:i+2048]
            h, p = librosa.effects.hpss(seg)
            hp = np.mean(h ** 2)
            pp = np.mean(p ** 2)
            if pp > 0:
                hnr_frames.append(10 * np.log10(hp / (pp + 1e-8)))
        features.hnr_std = float(np.std(hnr_frames)) if hnr_frames else 0.0

        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.mfcc_means = [float(x) for x in np.mean(mfccs, axis=1)]
        features.mfcc_stds = [float(x) for x in np.std(mfccs, axis=1)]

        # 频谱特征
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.spectral_centroid_mean = float(np.mean(sc))
        features.spectral_centroid_std = float(np.std(sc))
        features.spectral_bandwidth_mean = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        features.spectral_flatness_mean = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        features.spectral_rolloff_mean = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        features.zero_crossing_rate_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        # 韵律特征
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        features.speech_rate = float(len(onsets) / max(features.total_duration, 0.1))

        # 停顿检测
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        threshold = np.mean(rms) * 0.1
        silence = rms < threshold
        pause_changes = np.diff(silence.astype(int))
        features.pause_count = int(np.sum(pause_changes == 1))
        features.pause_ratio = float(np.sum(silence) / len(silence))

        features.rms_mean = float(np.mean(rms))
        features.rms_std = float(np.std(rms))
        features.energy_std = float(np.std(y ** 2))

        voiced = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0] > threshold
        features.voiced_ratio = float(np.sum(voiced) / len(voiced))

        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.tempo = float(tempo)
        except:
            pass

        # 共振峰近似（用MFCC反推）
        if features.f0_mean > 0:
            features.formant_f1 = float(features.mfcc_means[0] * 100 + 500) if features.mfcc_means[0] != 0 else 500
            features.formant_f2 = float(features.mfcc_means[1] * 100 + 1500) if features.mfcc_means[1] != 0 else 1500
            features.formant_f3 = float(features.mfcc_means[2] * 100 + 2500) if features.mfcc_means[2] != 0 else 2500
            features.formant_f4 = float(features.mfcc_means[3] * 100 + 3500) if features.mfcc_means[3] != 0 else 3500

        return features

    def _extract_with_numpy(self, path: str) -> AcousticFeatures:
        """降级模式：纯numpy提取基础特征"""
        if HAS_SF:
            y, sr = sf.read(path)
            if y.ndim > 1:
                y = y.mean(axis=1)
        else:
            import wave, struct
            with wave.open(path, 'r') as w:
                sr = w.getframerate()
                frames = w.readframes(w.getnframes())
                y = np.array(struct.unpack(f'<{w.getnframes()}h', frames), dtype=np.float32) / 32768.0

        features = AcousticFeatures()
        features.total_duration = len(y) / sr
        features.rms_mean = float(np.sqrt(np.mean(y ** 2)))
        features.rms_std = float(np.std(np.sqrt(np.abs(y))))
        features.energy_std = float(np.std(y ** 2))

        # 零交叉率
        zcr = np.sum(np.abs(np.diff(np.sign(y)))) / (2 * len(y))
        features.zero_crossing_rate_mean = float(zcr)

        return features


def extract_features(audio_path: str, sr: int = 16000) -> AcousticFeatures:
    """便捷函数：提取声学特征"""
    ext = FeatureExtractor(sr=sr)
    return ext.extract(audio_path)


__all__ = ["FeatureExtractor", "AcousticFeatures", "extract_features"]
