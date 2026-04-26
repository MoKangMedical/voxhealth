"""
VoxHealth -- MIMO AI Health Insight Engine

Calls Xiaomi MIMO API to generate personalized,
warm, actionable health insights from voice detection results.
"""

import os
import httpx
import json
from typing import Dict, List, Optional

MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
MIMO_MODEL = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")


class HealthInsightGenerator:
    """
    AI Health Insight Generator

    Transforms cold detection data into warm health advice
    """

    def __init__(self, api_key: str = MIMO_API_KEY, base_url: str = MIMO_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url

    async def generate_insight(self, report: dict, user_name: str = "you") -> str:
        """
        Generate AI insight from health report

        Args:
            report: Full health report dict
            user_name: User name for personalization

        Returns:
            Natural language health insight
        """
        overall = report.get("overall_risk_level", "normal")
        score = report.get("overall_score", 0)
        diseases = report.get("diseases", [])
        features = report.get("feature_summary", {})

        high_risks = [d for d in diseases if d.get("risk_level") == "high"]
        medium_risks = [d for d in diseases if d.get("risk_level") == "medium"]

        system_prompt = """You are a warm and professional health advisor AI.
Your task is to provide personalized health insights based on voice biomarker detection results.
Be warm, non-alarming, and give specific actionable advice.
Do NOT make medical diagnoses. Always remind users this is for reference only.
Reply in 150-250 characters, in Chinese."""

        user_prompt = f"""User {user_name} voice health detection results:

Overall: {overall} (Score {score}/100)

Acoustic Features:
{chr(10).join(f'- {k}: {v}' for k, v in features.items())}

Detected Disease Risks:
"""
        for d in diseases[:8]:
            level = {"high": "HIGH", "medium": "MED", "low": "LOW"}.get(d.get("risk_level", ""), "?")
            user_prompt += f"- [{level}] {d['disease']}: {d['risk_score']:.0f}/100\n"

        if not self.api_key:
            return self._fallback_insight(report)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MIMO_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[VoxHealth] MIMO API error: {e}")
            return self._fallback_insight(report)

    def _fallback_insight(self, report: dict) -> str:
        """Fallback insight when API is unavailable"""
        overall = report.get("overall_risk_level", "normal")
        score = report.get("overall_score", 0)
        diseases = report.get("diseases", [])
        high_risks = [d for d in diseases if d.get("risk_level") == "high"]
        medium_risks = [d for d in diseases if d.get("risk_level") == "medium"]

        parts = []
        if overall == "normal":
            parts.append("Your voice analysis shows overall good health status.")
        elif overall == "slight attention":
            parts.append("Your voice analysis shows some indicators that need attention.")
        else:
            parts.append("Your voice analysis shows some areas that need attention.")

        if high_risks:
            names = ", ".join(d["disease"] for d in high_risks[:2])
            parts.append(f"Key areas: {names} show higher risk levels, recommend professional consultation.")
        if medium_risks:
            names = ", ".join(d["disease"] for d in medium_risks[:2])
            parts.append(f"{names} are at moderate levels, recommend regular monitoring.")

        parts.append("This report is for reference only, not a medical diagnosis.")
        return " ".join(parts)

    async def generate_trend_summary(self, trends: dict, user_name: str = "you") -> str:
        """Generate trend analysis summary"""
        if not trends:
            return "No trend data available yet. Continue recording to build your health timeline."

        if not self.api_key:
            return self._fallback_trend(trends)

        trend_text = ""
        for disease, points in trends.items():
            if len(points) >= 2:
                recent = points[-1]["value"]
                prev = points[-2]["value"]
                direction = "up" if recent > prev else "down"
                trend_text += f"- {disease}: {prev:.0f} -> {recent:.0f} ({direction})\n"

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MIMO_MODEL,
                        "messages": [
                            {"role": "system", "content": "Analyze health trends and provide a brief summary in Chinese, 100-150 chars."},
                            {"role": "user", "content": f"User {user_name} health trends:\n{trend_text}"},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 300,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._fallback_trend(trends)

    def _fallback_trend(self, trends: dict) -> str:
        """Fallback trend summary"""
        count = sum(len(v) for v in trends.values())
        diseases = len(trends)
        return f"Tracked {count} data points across {diseases} health indicators. Continue regular testing for better trend analysis."


insight_generator = HealthInsightGenerator()
