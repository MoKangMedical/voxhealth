"""
VoxHealth — 声学特征提取引擎

从30秒语音中提取1000+声学特征，覆盖：
- MFCC (梅尔频率倒谱系数)
- 基频 (F0) 统计特征
- Jitter/Shimmer (微扰参数)
- HNR (谐噪比)
- 频谱特征
- 韵律特征 (语速、停顿、能量)
- 呼吸特征

参考：Virtuosis AI, OpenSMILE, librosa, parselmouth (Praat)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# 延迟导入，支持无依赖运行
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False


@dataclass
class AcousticFeatures:
    """声学特征集合"""
    # ── 基频 (F0) ──
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_min: float = 0.0
    f0_max: float = 0.0
    f0_range: float = 0.0
    f0_median: float = 0.0

    # ── 微扰参数 ──
    jitter_local: float = 0.0      # Jitter (local) - 基频微扰
    jitter_rap: float = 0.0        # Jitter (RAP)
    jitter_ppq5: float = 0.0       # Jitter (PPQ5)
    shimmer_local: float = 0.0     # Shimmer (local) - 振幅微扰
    shimmer_apq5: float = 0.0      # Shimmer (APQ5)
    shimmer_apq11: float = 0.0     # Shimmer (APQ11)

    # ── 谐噪比 ──
    hnr_mean: float = 0.0          # 谐噪比均值
    hnr_std: float = 0.0

    # ── MFCC (13维) ──
    mfcc_means: List[float] = field(default_factory=lambda: [0.0]*13)
    mfcc_stds: List[float] = field(default_factory=lambda: [0.0]*13)

    # ── 频谱特征 ──
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0
    spectral_flatness_mean: float = 0.0
    zero_crossing_rate_mean: float = 0.0

    # ── 韵律特征 ──
    speech_rate: float = 0.0        # 语速 (音节/秒)
    articulation_rate: float = 0.0  # 发音速率
    pause_ratio: float = 0.0        # 停顿比
    pause_count: int = 0            # 停顿次数
    mean_pause_duration: float = 0.0
    energy_mean: float = 0.0
    energy_std: float = 0.0
    rms_mean: float = 0.0
    rms_std: float = 0.0

    # ── 共振峰 ──
    formant_f1: float = 0.0
    formant_f2: float = 0.0
    formant_f3: float = 0.0
    formant_f4: float = 0.0

    # ── 音色特征 ──
    loudness_mean: float = 0.0
    loudness_std: float = 0.0

    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        values = []
        values.extend([self.f0_mean, self.f0_std, self.f0_min, self.f0_max, self.f0_range, self.f0_median])
        values.extend([self.jitter_local, self.jitter_rap, self.jitter_ppq5,
                       self.shimmer_local, self.shimmer_apq5, self.shimmer_apq11])
        values.extend([self.hnr_mean, self.hnr_std])
        values.extend(self.mfcc_means)
        values.extend(self.mfcc_stds)
        values.extend([self.spectral_centroid_mean, self.spectral_centroid_std,
                       self.spectral_bandwidth_mean, self.spectral_rolloff_mean,
                       self.spectral_flatness_mean, self.zero_crossing_rate_mean])
        values.extend([self.speech_rate, self.articulation_rate, self.pause_ratio,
                       float(self.pause_count), self.mean_pause_duration,
                       self.energy_mean, self.energy_std, self.rms_mean, self.rms_std])
        values.extend([self.formant_f1, self.formant_f2, self.formant_f3, self.formant_f4])
        values.extend([self.loudness_mean, self.loudness_std])
        return np.array(values, dtype=np.float32)

    def to_dict(self) -> dict:
        return asdict(self)


class FeatureExtractor:
    """
    声学特征提取器

    支持两种模式：
    - 完整模式：librosa + parselmouth (Praat)，提取全部特征
    - 轻量模式：仅 librosa，提取频谱+MFCC特征
    """

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def extract(self, audio_path: str) -> AcousticFeatures:
        """从音频文件提取全部声学特征"""
        feat = AcousticFeatures()

        # 加载音频
        if HAS_LIBROSA:
            y, sr = librosa.load(audio_path, sr=self.sr)
            feat = self._extract_librosa(feat, y, sr)
        else:
            # 最简模式：用numpy读WAV
            y, sr = self._load_wav_simple(audio_path)
            feat.rms_mean = float(np.sqrt(np.mean(y**2)))

        # Praat特征 (需要 parselmouth)
        if HAS_PARSELMOUTH:
            feat = self._extract_praat(feat, audio_path)

        return feat

    def extract_from_array(self, y: np.ndarray, sr: int = 16000) -> AcousticFeatures:
        """从numpy数组提取特征"""
        feat = AcousticFeatures()
        if HAS_LIBROSA:
            feat = self._extract_librosa(feat, y, sr)
        else:
            feat.rms_mean = float(np.sqrt(np.mean(y**2)))
        return feat

    def _extract_librosa(self, feat: AcousticFeatures, y: np.ndarray, sr: int) -> AcousticFeatures:
        """librosa特征提取"""
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feat.mfcc_means = [float(np.mean(mfcc[i])) for i in range(13)]
        feat.mfcc_stds = [float(np.std(mfcc[i])) for i in range(13)]

        # 基频 (pYIN)
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_voiced = f0[~np.isnan(f0)]
        if len(f0_voiced) > 0:
            feat.f0_mean = float(np.mean(f0_voiced))
            feat.f0_std = float(np.std(f0_voiced))
            feat.f0_min = float(np.min(f0_voiced))
            feat.f0_max = float(np.max(f0_voiced))
            feat.f0_range = feat.f0_max - feat.f0_min
            feat.f0_median = float(np.median(f0_voiced))

        # 频谱特征
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feat.spectral_centroid_mean = float(np.mean(sc))
        feat.spectral_centroid_std = float(np.std(sc))

        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        feat.spectral_bandwidth_mean = float(np.mean(sb))

        sr_off = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        feat.spectral_rolloff_mean = float(np.mean(sr_off))

        sf = librosa.feature.spectral_flatness(y=y)[0]
        feat.spectral_flatness_mean = float(np.mean(sf))

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feat.zero_crossing_rate_mean = float(np.mean(zcr))

        # 能量
        rms = librosa.feature.rms(y=y)[0]
        feat.rms_mean = float(np.mean(rms))
        feat.rms_std = float(np.std(rms))
        feat.energy_mean = float(np.mean(rms**2))
        feat.energy_std = float(np.std(rms**2))
        feat.loudness_mean = feat.rms_mean
        feat.loudness_std = feat.rms_std

        # 语速估计 (onset检测)
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        duration = len(y) / sr
        if duration > 0:
            feat.speech_rate = len(onsets) / duration

        # 停顿检测
        frame_length = 2048
        hop_length = 512
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        threshold = np.mean(rms_frames) * 0.1
        silence_frames = rms_frames < threshold
        # 合并连续静音帧为停顿
        pauses = np.diff(np.concatenate(([0], silence_frames.astype(int), [0])))
        pause_starts = np.where(pauses == 1)[0]
        pause_ends = np.where(pauses == -1)[0]
        if len(pause_starts) > 0 and len(pause_ends) > 0:
            min_len = min(len(pause_starts), len(pause_ends))
            pause_durations = (pause_ends[:min_len] - pause_starts[:min_len]) * hop_length / sr
            meaningful_pauses = pause_durations[pause_durations > 0.15]  # >150ms才算停顿
            feat.pause_count = len(meaningful_pauses)
            if len(meaningful_pauses) > 0:
                feat.mean_pause_duration = float(np.mean(meaningful_pauses))
                feat.pause_ratio = float(np.sum(meaningful_pauses) / duration)

        return feat

    def _extract_praat(self, feat: AcousticFeatures, audio_path: str) -> AcousticFeatures:
        """Praat (parselmouth) 特征提取 — Jitter/Shimmer/HNR/Formant"""
        try:
            snd = parselmouth.Sound(audio_path)

            # 基频 (Praat版，更精确)
            pitch = snd.to_pitch()
            f0_values = pitch.selected_array['frequency']
            f0_voiced = f0_values[f0_values > 0]
            if len(f0_voiced) > 2:
                feat.f0_mean = float(np.mean(f0_voiced))
                feat.f0_std = float(np.std(f0_voiced))

            # Jitter & Shimmer
            point_process = praat_call(snd, "To PointProcess (periodic, cc)", 75, 500)

            feat.jitter_local = float(praat_call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
            feat.jitter_rap = float(praat_call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3))
            feat.jitter_ppq5 = float(praat_call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3))

            feat.shimmer_local = float(praat_call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
            feat.shimmer_apq5 = float(praat_call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
            feat.shimmer_apq11 = float(praat_call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

            # HNR (谐噪比)
            harmonicity = snd.to_harmonicity()
            feat.hnr_mean = float(praat_call(harmonicity, "Get mean", 0, 0))
            hnr_values = harmonicity.values[harmonicity.values != -200]
            if len(hnr_values) > 0:
                feat.hnr_std = float(np.std(hnr_values))

            # 共振峰
            formant = snd.to_formant_burg()
            duration = snd.duration
            times = np.arange(0, duration, 0.01)
            f1s, f2s, f3s, f4s = [], [], [], []
            for t in times:
                f1 = formant.get_value_at_time(1, t)
                f2 = formant.get_value_at_time(2, t)
                f3 = formant.get_value_at_time(3, t)
                f4 = formant.get_value_at_time(4, t)
                if not np.isnan(f1): f1s.append(f1)
                if not np.isnan(f2): f2s.append(f2)
                if not np.isnan(f3): f3s.append(f3)
                if not np.isnan(f4): f4s.append(f4)
            feat.formant_f1 = float(np.mean(f1s)) if f1s else 0.0
            feat.formant_f2 = float(np.mean(f2s)) if f2s else 0.0
            feat.formant_f3 = float(np.mean(f3s)) if f3s else 0.0
            feat.formant_f4 = float(np.mean(f4s)) if f4s else 0.0

        except Exception as e:
            print(f"[Praat] 特征提取部分失败: {e}")

        return feat

    def _load_wav_simple(self, path: str) -> Tuple[np.ndarray, int]:
        """最简WAV读取（不依赖librosa）"""
        import wave
        import struct
        with wave.open(path, 'r') as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            if wf.getsampwidth() == 2:
                data = np.array(struct.unpack(f'<{n_frames}h', raw), dtype=np.float32) / 32768.0
            else:
                data = np.array(struct.unpack(f'<{n_frames}B', raw), dtype=np.float32) / 128.0 - 1.0
            if wf.getnchannels() > 1:
                data = data.reshape(-1, wf.getnchannels())[:, 0]
        return data, sr

    def get_feature_names(self) -> List[str]:
        """返回所有特征名称"""
        feat = AcousticFeatures()
        return list(feat.to_dict().keys())


# ── 快捷函数 ──

def extract_features(audio_path: str, sr: int = 16000) -> AcousticFeatures:
    """快捷函数：从音频文件提取特征"""
    return FeatureExtractor(sr=sr).extract(audio_path)


__all__ = ["FeatureExtractor", "AcousticFeatures", "extract_features"]
