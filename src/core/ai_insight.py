"""
VoxHealth — MIMO AI 健康解读引擎

调用小米MIMO API，为用户的语音检测结果生成
个性化的、温暖的、可操作的健康解读。
"""

import os
import httpx
import json
from typing import Dict, List, Optional

MIMO_API_KEY = os.getenv("MIMO_API_KEY", "sk-ccwzuzw9e1t42xjok84nfx7wrv4geuzc590ojipwfqga5uxl")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")


class HealthInsightGenerator:
    """
    AI健康解读生成器

    将冰冷的检测数据转化为温暖的健康建议
    """

    def __init__(self, api_key: str = MIMO_API_KEY, base_url: str = MIMO_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url

    async def generate_insight(self, report: dict, user_name: str = "您") -> str:
        """
        根据健康报告生成AI解读

        Args:
            report: 完整健康报告dict
            user_name: 用户称呼

        Returns:
            自然语言健康解读
        """
        # 提取关键信息
        overall = report.get("overall_risk_level", "正常")
        score = report.get("overall_score", 0)
        diseases = report.get("diseases", [])
        features = report.get("feature_summary", {})

        # 高风险疾病
        high_risks = [d for d in diseases if d.get("risk_level") == "高"]
        medium_risks = [d for d in diseases if d.get("risk_level") == "中"]

        # 构建prompt
        system_prompt = """你是一位温暖且专业的健康顾问AI。
你的任务是根据语音生物标志物检测结果，为用户提供个性化的健康解读。
语气要温暖、不吓人、给出具体可行的建议。
不要做医学诊断，始终提醒用户这仅供参考。
回复长度：150-250字，用中文。"""

        user_prompt = f"""用户{user_name}的语音健康检测结果：

总体评估：{overall}（评分 {score}/100）

声学特征：
{chr(10).join(f'- {k}: {v}' for k, v in features.items())}

检测到的疾病风险：
"""
        for d in diseases[:8]:
            level_emoji = {"高": "🔴", "中": "🟡", "低": "🟢"}.get(d.get("risk_level", ""), "⚪")
            user_prompt += f"- {level_emoji} {d['disease']}: {d.get('risk_score', 0)}分 ({d.get('risk_level', '')}) - 关键标志: {', '.join(d.get('key_markers', [])[:2])}\n"

        if high_risks:
            user_prompt += f"\n重点关注：{', '.join(d['disease'] for d in high_risks)}\n"

        user_prompt += "\n请为用户生成一段温暖、专业的健康解读和建议。"

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "mimo-v2-pro",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500
                    }
                )
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            return self._fallback_insight(report, user_name)

    async def generate_trend_summary(self, trends: dict, user_name: str = "您") -> str:
        """生成趋势总结"""
        if not trends:
            return f"{user_name}目前还没有足够的历史数据来生成趋势分析。建议定期检测，积累数据后可以看到变化趋势。"

        system_prompt = "你是一位健康数据分析AI。根据用户的健康趋势数据，生成简短的趋势总结（100字内）。"

        trend_text = ""
        for metric, points in list(trends.items())[:5]:
            if len(points) >= 2:
                change = points[-1]["value"] - points[0]["value"]
                direction = "上升↑" if change > 0 else "下降↓"
                trend_text += f"- {metric}: {direction} ({points[0]['value']:.0f}→{points[-1]['value']:.0f})\n"

        user_prompt = f"{user_name}的健康趋势数据（最近30天）：\n{trend_text}\n请生成趋势总结。"

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "mimo-v2-pro",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.5,
                        "max_tokens": 300
                    }
                )
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except:
            return "趋势分析暂时不可用，请稍后重试。"

    def _fallback_insight(self, report: dict, user_name: str) -> str:
        """备用解读（API不可用时）"""
        overall = report.get("overall_risk_level", "正常")
        diseases = report.get("diseases", [])
        high_risks = [d for d in diseases if d.get("risk_level") == "高"]
        medium_risks = [d for d in diseases if d.get("risk_level") == "中"]

        insight = f"{user_name}，您好！\n\n"
        insight += f"您的语音健康评估结果为「{overall}」。"

        if high_risks:
            names = "、".join(d["disease"] for d in high_risks[:3])
            insight += f"\n\n⚠️ 需要关注：{names}。建议您关注这些指标的变化，如有不适请咨询专业医生。"
        elif medium_risks:
            names = "、".join(d["disease"] for d in medium_risks[:3])
            insight += f"\n\n💡 轻微关注：{names}。这些指标处于中等水平，建议定期检测观察趋势。"
        else:
            insight += "\n\n✅ 各项指标均在正常范围内，请继续保持健康的生活方式。"

        insight += "\n\n📌 此报告基于AI语音分析，仅供参考，不构成医学诊断。"
        return insight


insight_generator = HealthInsightGenerator()
