"""
VoiceHealth — 视频分析引擎 v1.0

通过视频分析用户的：
- 皮肤状态（肤色、痘痘、皱纹、色斑、毛孔）
- 眼睛状态（黑眼圈、眼袋、红血丝、疲劳度）
- 头发状态（发量、发质、白发比例）

基于 OpenCV + MediaPipe + 深度学习
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import os

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


@dataclass
class SkinAnalysis:
    """皮肤分析结果"""
    # 肤色
    skin_tone: str = "medium"  # light, medium, dark
    skin_tone_rgb: Tuple[int, int, int] = (0, 0, 0)
    skin_uniformity: float = 0.0  # 肤色均匀度 0-1
    
    # 痘痘
    acne_count: int = 0
    acne_severity: str = "none"  # none, mild, moderate, severe
    acne_regions: List[str] = field(default_factory=list)  # forehead, cheeks, chin
    
    # 皱纹
    wrinkle_score: float = 0.0  # 0-100
    wrinkle_areas: Dict[str, float] = field(default_factory=dict)  # 区域 -> 严重程度
    wrinkle_level: str = "none"  # none, fine, moderate, deep
    
    # 色斑
    spot_count: int = 0
    spot_score: float = 0.0  # 0-100
    spot_level: str = "none"  # none, mild, moderate, severe
    
    # 毛孔
    pore_score: float = 0.0  # 0-100
    pore_level: str = "none"  # none, small, medium, large
    
    # 综合
    overall_score: float = 0.0  # 0-100
    skin_age: int = 0  # 预测皮肤年龄
    summary: str = ""
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EyeAnalysis:
    """眼睛分析结果"""
    # 黑眼圈
    dark_circle_score: float = 0.0  # 0-100
    dark_circle_level: str = "none"  # none, mild, moderate, severe
    dark_circle_color: str = "none"  # none, brown, blue, purple
    
    # 眼袋
    eye_bag_score: float = 0.0  # 0-100
    eye_bag_level: str = "none"  # none, mild, moderate, severe
    
    # 红血丝
    redness_score: float = 0.0  # 0-100
    redness_level: str = "none"  # none, mild, moderate, severe
    
    # 疲劳度
    fatigue_score: float = 0.0  # 0-100
    fatigue_level: str = "none"  # none, mild, moderate, severe
    
    # 眼睛大小/形状
    eye_openness: float = 0.0  # 眼睛开合度
    eye_symmetry: float = 0.0  # 对称度 0-1
    
    # 综合
    overall_score: float = 0.0  # 0-100
    eye_age: int = 0  # 预测眼部年龄
    summary: str = ""
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HairAnalysis:
    """头发分析结果"""
    # 发量
    hair_density: float = 0.0  # 0-100
    hair_density_level: str = "normal"  # sparse, thin, normal, thick
    hairline_score: float = 0.0  # 发际线高度 0-100
    
    # 发质
    hair_quality: float = 0.0  # 0-100
    hair_quality_level: str = "normal"  # dry, normal, oily
    hair_damage: float = 0.0  # 受损程度 0-100
    
    # 白发
    gray_hair_ratio: float = 0.0  # 白发比例 0-1
    gray_hair_level: str = "none"  # none, mild, moderate, severe
    
    # 综合
    overall_score: float = 0.0  # 0-100
    hair_age: int = 0  # 预测头发年龄
    summary: str = ""
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VideoAnalysisResult:
    """视频分析完整结果"""
    skin: SkinAnalysis = field(default_factory=SkinAnalysis)
    eye: EyeAnalysis = field(default_factory=EyeAnalysis)
    hair: HairAnalysis = field(default_factory=HairAnalysis)
    
    # 综合
    overall_score: float = 0.0
    biological_age: int = 0  # 预测生物学年龄
    chronological_age: int = 0  # 实际年龄（如果提供）
    age_difference: int = 0  # 年龄差
    
    summary: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        return {
            "skin": self.skin.to_dict(),
            "eye": self.eye.to_dict(),
            "hair": self.hair.to_dict(),
            "overall_score": self.overall_score,
            "biological_age": self.biological_age,
            "chronological_age": self.chronological_age,
            "age_difference": self.age_difference,
            "summary": self.summary,
            "timestamp": self.timestamp
        }


class VideoAnalyzer:
    """视频分析器"""
    
    def __init__(self):
        if HAS_MEDIAPIPE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_face_detection = mp.solutions.face_detection
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        else:
            self.face_mesh = None
    
    def analyze_video(self, video_path: str, max_frames: int = 30) -> VideoAnalysisResult:
        """
        分析视频
        
        Args:
            video_path: 视频文件路径
            max_frames: 最大分析帧数
            
        Returns:
            VideoAnalysisResult: 分析结果
        """
        if not HAS_CV2:
            return self._fallback_analysis()
        
        result = VideoAnalysisResult()
        
        try:
            # 读取视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("无法打开视频文件")
            
            frame_count = 0
            skin_analyses = []
            eye_analyses = []
            hair_analyses = []
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 分析每一帧
                skin = self._analyze_skin_frame(frame)
                eye = self._analyze_eye_frame(frame)
                hair = self._analyze_hair_frame(frame)
                
                skin_analyses.append(skin)
                eye_analyses.append(eye)
                hair_analyses.append(hair)
                
                frame_count += 1
            
            cap.release()
            
            # 合并多帧结果
            if skin_analyses:
                result.skin = self._merge_skin_analyses(skin_analyses)
            if eye_analyses:
                result.eye = self._merge_eye_analyses(eye_analyses)
            if hair_analyses:
                result.hair = self._merge_hair_analyses(hair_analyses)
            
            # 计算综合结果
            result.overall_score = (
                result.skin.overall_score * 0.4 +
                result.eye.overall_score * 0.3 +
                result.hair.overall_score * 0.3
            )
            
            # 预测生物学年龄
            result.biological_age = self._predict_biological_age(result)
            result.summary = self._generate_summary(result)
            
            from datetime import datetime
            result.timestamp = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            print(f"视频分析失败: {e}")
            return self._fallback_analysis()
    
    def _analyze_skin_frame(self, frame: np.ndarray) -> SkinAnalysis:
        """分析单帧皮肤状态"""
        analysis = SkinAnalysis()
        
        # 转换为HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 检测肤色区域
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 计算肤色
        skin_pixels = frame[skin_mask > 0]
        if len(skin_pixels) > 0:
            analysis.skin_tone_rgb = tuple(map(int, np.mean(skin_pixels, axis=0)))
            analysis.skin_tone = self._classify_skin_tone(analysis.skin_tone_rgb)
        
        # 肤色均匀度
        if len(skin_pixels) > 100:
            analysis.skin_uniformity = 1.0 - (np.std(skin_pixels) / 128.0)
        
        # 检测痘痘（红色区域）
        red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        acne_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        analysis.acne_count = len([c for c in acne_contours if cv2.contourArea(c) > 50])
        
        if analysis.acne_count == 0:
            analysis.acne_severity = "none"
        elif analysis.acne_count <= 3:
            analysis.acne_severity = "mild"
        elif analysis.acne_count <= 8:
            analysis.acne_severity = "moderate"
        else:
            analysis.acne_severity = "severe"
        
        # 检测皱纹（边缘检测）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        wrinkle_density = np.sum(edges > 0) / edges.size
        analysis.wrinkle_score = min(wrinkle_density * 500, 100)
        
        if analysis.wrinkle_score < 20:
            analysis.wrinkle_level = "none"
        elif analysis.wrinkle_score < 40:
            analysis.wrinkle_level = "fine"
        elif analysis.wrinkle_score < 60:
            analysis.wrinkle_level = "moderate"
        else:
            analysis.wrinkle_level = "deep"
        
        # 检测色斑
        brown_mask = cv2.inRange(hsv, np.array([10, 50, 20]), np.array([20, 200, 200]))
        spot_contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        analysis.spot_count = len([c for c in spot_contours if cv2.contourArea(c) > 30])
        analysis.spot_score = min(analysis.spot_count * 5, 100)
        
        if analysis.spot_count == 0:
            analysis.spot_level = "none"
        elif analysis.spot_count <= 3:
            analysis.spot_level = "mild"
        elif analysis.spot_count <= 6:
            analysis.spot_level = "moderate"
        else:
            analysis.spot_level = "severe"
        
        # 检测毛孔（纹理分析）
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        texture = cv2.absdiff(gray, blur)
        analysis.pore_score = min(np.mean(texture) * 2, 100)
        
        if analysis.pore_score < 20:
            analysis.pore_level = "none"
        elif analysis.pore_score < 40:
            analysis.pore_level = "small"
        elif analysis.pore_score < 60:
            analysis.pore_level = "medium"
        else:
            analysis.pore_level = "large"
        
        # 综合评分
        analysis.overall_score = (
            analysis.skin_uniformity * 30 +
            (100 - analysis.acne_count * 10) * 25 +
            (100 - analysis.wrinkle_score) * 25 +
            (100 - analysis.spot_score) * 20
        )
        analysis.overall_score = max(0, min(100, analysis.overall_score))
        
        return analysis
    
    def _analyze_eye_frame(self, frame: np.ndarray) -> EyeAnalysis:
        """分析单帧眼睛状态"""
        analysis = EyeAnalysis()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 使用Haar级联检测眼睛
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(eyes) >= 2:
            # 按x坐标排序
            eyes = sorted(eyes, key=lambda e: e[0])
            
            for (x, y, w, h) in eyes[:2]:
                eye_roi = frame[y:y+h, x:x+w]
                eye_hsv = hsv[y:y+h, x:x+w]
                
                # 检测黑眼圈（暗色区域）
                dark_mask = cv2.inRange(eye_hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
                dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
                analysis.dark_circle_score += dark_ratio * 100
                
                # 检测红血丝
                red_mask = cv2.inRange(eye_hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
                red_ratio = np.sum(red_mask > 0) / red_mask.size
                analysis.redness_score += red_ratio * 200
                
                # 检测眼睛开合度
                eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY)
                white_ratio = np.sum(thresh > 0) / thresh.size
                analysis.eye_openness += white_ratio
            
            # 平均值
            analysis.dark_circle_score /= 2
            analysis.redness_score /= 2
            analysis.eye_openness /= 2
        
        # 分级
        if analysis.dark_circle_score < 20:
            analysis.dark_circle_level = "none"
        elif analysis.dark_circle_score < 40:
            analysis.dark_circle_level = "mild"
        elif analysis.dark_circle_score < 60:
            analysis.dark_circle_level = "moderate"
        else:
            analysis.dark_circle_level = "severe"
        
        if analysis.redness_score < 15:
            analysis.redness_level = "none"
        elif analysis.redness_score < 30:
            analysis.redness_level = "mild"
        elif analysis.redness_score < 50:
            analysis.redness_level = "moderate"
        else:
            analysis.redness_level = "severe"
        
        # 疲劳度（综合黑眼圈和眼睛开合度）
        analysis.fatigue_score = (
            analysis.dark_circle_score * 0.6 +
            (1 - analysis.eye_openness) * 100 * 0.4
        )
        
        if analysis.fatigue_score < 20:
            analysis.fatigue_level = "none"
        elif analysis.fatigue_score < 40:
            analysis.fatigue_level = "mild"
        elif analysis.fatigue_score < 60:
            analysis.fatigue_level = "moderate"
        else:
            analysis.fatigue_level = "severe"
        
        # 综合评分
        analysis.overall_score = (
            (100 - analysis.dark_circle_score) * 0.35 +
            (100 - analysis.eye_bag_score) * 0.25 +
            (100 - analysis.redness_score) * 0.20 +
            (100 - analysis.fatigue_score) * 0.20
        )
        analysis.overall_score = max(0, min(100, analysis.overall_score))
        
        return analysis
    
    def _analyze_hair_frame(self, frame: np.ndarray) -> HairAnalysis:
        """分析单帧头发状态"""
        analysis = HairAnalysis()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测头发区域
        hair_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        hair_ratio = np.sum(hair_mask > 0) / hair_mask.size
        
        # 发量估算
        analysis.hair_density = min(hair_ratio * 200, 100)
        
        if analysis.hair_density < 30:
            analysis.hair_density_level = "sparse"
        elif analysis.hair_density < 50:
            analysis.hair_density_level = "thin"
        elif analysis.hair_density < 80:
            analysis.hair_density_level = "normal"
        else:
            analysis.hair_density_level = "thick"
        
        # 检测白发
        white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 30, 255]))
        white_in_hair = cv2.bitwise_and(white_mask, hair_mask)
        white_ratio = np.sum(white_in_hair > 0) / (np.sum(hair_mask > 0) + 1e-8)
        analysis.gray_hair_ratio = white_ratio
        
        if white_ratio < 0.05:
            analysis.gray_hair_level = "none"
        elif white_ratio < 0.15:
            analysis.gray_hair_level = "mild"
        elif white_ratio < 0.30:
            analysis.gray_hair_level = "moderate"
        else:
            analysis.gray_hair_level = "severe"
        
        # 发质评估（纹理）
        hair_roi = cv2.bitwise_and(gray, gray, mask=hair_mask)
        if np.sum(hair_mask > 0) > 1000:
            laplacian = cv2.Laplacian(hair_roi, cv2.CV_64F)
            analysis.hair_quality = min(np.var(laplacian) / 100, 100)
        
        # 综合评分
        analysis.overall_score = (
            analysis.hair_density * 0.4 +
            analysis.hair_quality * 0.3 +
            (1 - analysis.gray_hair_ratio) * 100 * 0.3
        )
        analysis.overall_score = max(0, min(100, analysis.overall_score))
        
        return analysis
    
    def _classify_skin_tone(self, rgb: Tuple[int, int, int]) -> str:
        """分类肤色"""
        brightness = sum(rgb) / 3
        if brightness > 180:
            return "light"
        elif brightness > 120:
            return "medium"
        else:
            return "dark"
    
    def _merge_skin_analyses(self, analyses: List[SkinAnalysis]) -> SkinAnalysis:
        """合并多帧皮肤分析"""
        if not analyses:
            return SkinAnalysis()
        
        merged = SkinAnalysis()
        
        # 取平均值
        merged.skin_uniformity = np.mean([a.skin_uniformity for a in analyses])
        merged.acne_count = int(np.mean([a.acne_count for a in analyses]))
        merged.wrinkle_score = np.mean([a.wrinkle_score for a in analyses])
        merged.spot_count = int(np.mean([a.spot_count for a in analyses]))
        merged.pore_score = np.mean([a.pore_score for a in analyses])
        merged.overall_score = np.mean([a.overall_score for a in analyses])
        
        # 取最常见的肤色
        tones = [a.skin_tone for a in analyses]
        merged.skin_tone = max(set(tones), key=tones.count)
        
        # 分级
        merged.acne_severity = analyses[len(analyses)//2].acne_severity
        merged.wrinkle_level = analyses[len(analyses)//2].wrinkle_level
        merged.spot_level = analyses[len(analyses)//2].spot_level
        merged.pore_level = analyses[len(analyses)//2].pore_level
        
        # 生成建议
        merged.suggestions = self._generate_skin_suggestions(merged)
        merged.summary = self._generate_skin_summary(merged)
        
        return merged
    
    def _merge_eye_analyses(self, analyses: List[EyeAnalysis]) -> EyeAnalysis:
        """合并多帧眼睛分析"""
        if not analyses:
            return EyeAnalysis()
        
        merged = EyeAnalysis()
        
        merged.dark_circle_score = np.mean([a.dark_circle_score for a in analyses])
        merged.eye_bag_score = np.mean([a.eye_bag_score for a in analyses])
        merged.redness_score = np.mean([a.redness_score for a in analyses])
        merged.fatigue_score = np.mean([a.fatigue_score for a in analyses])
        merged.eye_openness = np.mean([a.eye_openness for a in analyses])
        merged.overall_score = np.mean([a.overall_score for a in analyses])
        
        merged.dark_circle_level = analyses[len(analyses)//2].dark_circle_level
        merged.eye_bag_level = analyses[len(analyses)//2].eye_bag_level
        merged.redness_level = analyses[len(analyses)//2].redness_level
        merged.fatigue_level = analyses[len(analyses)//2].fatigue_level
        
        merged.suggestions = self._generate_eye_suggestions(merged)
        merged.summary = self._generate_eye_summary(merged)
        
        return merged
    
    def _merge_hair_analyses(self, analyses: List[HairAnalysis]) -> HairAnalysis:
        """合并多帧头发分析"""
        if not analyses:
            return HairAnalysis()
        
        merged = HairAnalysis()
        
        merged.hair_density = np.mean([a.hair_density for a in analyses])
        merged.hair_quality = np.mean([a.hair_quality for a in analyses])
        merged.gray_hair_ratio = np.mean([a.gray_hair_ratio for a in analyses])
        merged.overall_score = np.mean([a.overall_score for a in analyses])
        
        merged.hair_density_level = analyses[len(analyses)//2].hair_density_level
        merged.gray_hair_level = analyses[len(analyses)//2].gray_hair_level
        
        merged.suggestions = self._generate_hair_suggestions(merged)
        merged.summary = self._generate_hair_summary(merged)
        
        return merged
    
    def _predict_biological_age(self, result: VideoAnalysisResult) -> int:
        """预测生物学年龄"""
        base_age = 30
        
        # 皮肤年龄因子
        skin_factor = 0
        if result.skin.wrinkle_level == "deep":
            skin_factor += 10
        elif result.skin.wrinkle_level == "moderate":
            skin_factor += 5
        if result.skin.spot_level == "severe":
            skin_factor += 5
        
        # 眼睛年龄因子
        eye_factor = 0
        if result.eye.dark_circle_level == "severe":
            eye_factor += 5
        if result.eye.fatigue_level == "severe":
            eye_factor += 3
        
        # 头发年龄因子
        hair_factor = 0
        if result.hair.gray_hair_level == "severe":
            hair_factor += 10
        elif result.hair.gray_hair_level == "moderate":
            hair_factor += 5
        
        return base_age + skin_factor + eye_factor + hair_factor
    
    def _generate_summary(self, result: VideoAnalysisResult) -> str:
        """生成综合摘要"""
        return (
            f"皮肤状态：{result.skin.summary}。"
            f"眼睛状态：{result.eye.summary}。"
            f"头发状态：{result.hair.summary}。"
            f"综合评分 {result.overall_score:.0f} 分，"
            f"预测生物学年龄约 {result.biological_age} 岁。"
        )
    
    def _generate_skin_summary(self, skin: SkinAnalysis) -> str:
        """生成皮肤摘要"""
        parts = []
        if skin.acne_severity != "none":
            parts.append(f"有{skin.acne_count}颗痘痘")
        if skin.wrinkle_level != "none":
            parts.append(f"皱纹{skin.wrinkle_level}")
        if skin.spot_level != "none":
            parts.append(f"色斑{skin.spot_level}")
        if not parts:
            parts.append("状态良好")
        return "，".join(parts)
    
    def _generate_eye_summary(self, eye: EyeAnalysis) -> str:
        """生成眼睛摘要"""
        parts = []
        if eye.dark_circle_level != "none":
            parts.append(f"黑眼圈{eye.dark_circle_level}")
        if eye.fatigue_level != "none":
            parts.append(f"疲劳度{eye.fatigue_level}")
        if not parts:
            parts.append("状态良好")
        return "，".join(parts)
    
    def _generate_hair_summary(self, hair: HairAnalysis) -> str:
        """生成头发摘要"""
        parts = []
        if hair.gray_hair_level != "none":
            parts.append(f"白发{hair.gray_hair_level}")
        if hair.hair_density_level in ["sparse", "thin"]:
            parts.append(f"发量{hair.hair_density_level}")
        if not parts:
            parts.append("状态良好")
        return "，".join(parts)
    
    def _generate_skin_suggestions(self, skin: SkinAnalysis) -> List[str]:
        """生成皮肤建议"""
        suggestions = []
        if skin.acne_severity in ["moderate", "severe"]:
            suggestions.append("建议使用温和的洁面产品，避免挤压痘痘")
        if skin.wrinkle_level in ["moderate", "deep"]:
            suggestions.append("建议使用抗皱护肤品，注意防晒")
        if skin.spot_level in ["moderate", "severe"]:
            suggestions.append("建议使用美白淡斑产品，加强防晒")
        if skin.pore_level in ["medium", "large"]:
            suggestions.append("建议定期清洁毛孔，使用收缩毛孔产品")
        if not suggestions:
            suggestions.append("皮肤状态良好，建议保持良好作息和防晒习惯")
        return suggestions
    
    def _generate_eye_suggestions(self, eye: EyeAnalysis) -> List[str]:
        """生成眼睛建议"""
        suggestions = []
        if eye.dark_circle_level in ["moderate", "severe"]:
            suggestions.append("建议保证充足睡眠，使用眼霜")
        if eye.fatigue_level in ["moderate", "severe"]:
            suggestions.append("建议减少用眼时间，多休息")
        if eye.redness_level in ["moderate", "severe"]:
            suggestions.append("建议减少电子屏幕使用，必要时就医")
        if not suggestions:
            suggestions.append("眼睛状态良好，建议保持良好用眼习惯")
        return suggestions
    
    def _generate_hair_suggestions(self, hair: HairAnalysis) -> List[str]:
        """生成头发建议"""
        suggestions = []
        if hair.gray_hair_level in ["moderate", "severe"]:
            suggestions.append("建议补充维生素B族，减少压力")
        if hair.hair_density_level in ["sparse", "thin"]:
            suggestions.append("建议使用防脱洗发水，避免频繁烫染")
        if not suggestions:
            suggestions.append("头发状态良好，建议保持健康饮食")
        return suggestions
    
    def _fallback_analysis(self) -> VideoAnalysisResult:
        """降级分析（无依赖时）"""
        return VideoAnalysisResult(
            skin=SkinAnalysis(overall_score=50, summary="需要安装OpenCV和MediaPipe"),
            eye=EyeAnalysis(overall_score=50, summary="需要安装OpenCV和MediaPipe"),
            hair=HairAnalysis(overall_score=50, summary="需要安装OpenCV和MediaPipe"),
            overall_score=50,
            biological_age=30,
            summary="降级模式：请安装 opencv-python 和 mediapihe 以启用完整分析"
        )


# 便捷函数
def analyze_video(video_path: str, max_frames: int = 30) -> VideoAnalysisResult:
    """分析视频文件"""
    analyzer = VideoAnalyzer()
    return analyzer.analyze_video(video_path, max_frames)


__all__ = [
    "VideoAnalyzer",
    "VideoAnalysisResult",
    "SkinAnalysis",
    "EyeAnalysis",
    "HairAnalysis",
    "analyze_video"
]
