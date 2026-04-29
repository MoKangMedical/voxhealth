"""
VoiceHealth — 数据库层

用户系统 + 健康记录 + 检测历史
SQLite存储，轻量部署
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

DB_PATH = os.getenv("VOICEHEALTH_DB", str(Path(__file__).parent.parent.parent / "data" / "voicehealth.db"))


class Database:
    """VoiceHealth数据库管理"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            phone TEXT UNIQUE,
            nickname TEXT,
            gender TEXT DEFAULT '',
            age INTEGER DEFAULT 0,
            avatar_url TEXT DEFAULT '',
            created_at TEXT,
            last_login TEXT,
            health_goals TEXT DEFAULT '[]',
            settings TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS health_records (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            record_type TEXT DEFAULT 'voice_check',
            overall_score REAL DEFAULT 0,
            overall_level TEXT DEFAULT '',
            diseases_json TEXT DEFAULT '[]',
            features_json TEXT DEFAULT '{}',
            recommendations_json TEXT DEFAULT '[]',
            recording_quality TEXT DEFAULT '',
            ai_insight TEXT DEFAULT '',
            audio_path TEXT DEFAULT '',
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS health_trends (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            recorded_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE INDEX IF NOT EXISTS idx_records_user ON health_records(user_id);
        CREATE INDEX IF NOT EXISTS idx_records_date ON health_records(created_at);
        CREATE INDEX IF NOT EXISTS idx_trends_user ON health_trends(user_id);
        CREATE INDEX IF NOT EXISTS idx_trends_metric ON health_trends(metric_name);
        """)
        conn.commit()
        conn.close()

    # ── 用户管理 ──

    def create_user(self, phone: str, nickname: str = "", gender: str = "", age: int = 0) -> dict:
        """创建用户"""
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO users (id, phone, nickname, gender, age, created_at, last_login) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, phone, nickname or f"用户{phone[-4:]}", gender, age, now, now)
            )
            conn.commit()
            return {"id": user_id, "phone": phone, "nickname": nickname or f"用户{phone[-4:]}", "created_at": now}
        except sqlite3.IntegrityError:
            return self.get_user_by_phone(phone)
        finally:
            conn.close()

    def get_user(self, user_id: str) -> Optional[dict]:
        """获取用户"""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_user_by_phone(self, phone: str) -> Optional[dict]:
        """通过手机号获取用户"""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
        conn.close()
        if row:
            conn = self._get_conn()
            conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now().isoformat(), row["id"]))
            conn.commit()
            conn.close()
        return dict(row) if row else None

    def update_user(self, user_id: str, **kwargs) -> bool:
        """更新用户信息"""
        allowed = {"nickname", "gender", "age", "avatar_url", "health_goals", "settings"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        sets = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [user_id]
        conn = self._get_conn()
        conn.execute(f"UPDATE users SET {sets} WHERE id = ?", values)
        conn.commit()
        conn.close()
        return True

    # ── 健康记录 ──

    def save_health_record(self, user_id: str, report: dict, features: dict = None,
                           ai_insight: str = "", audio_path: str = "") -> str:
        """保存健康检测记录"""
        record_id = f"rec_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()
        conn = self._get_conn()

        diseases = report.get("diseases", [])
        recommendations = report.get("recommendations", [])

        conn.execute("""
            INSERT INTO health_records
            (id, user_id, overall_score, overall_level, diseases_json, features_json,
             recommendations_json, recording_quality, ai_insight, audio_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record_id, user_id,
            report.get("overall_score", 0),
            report.get("overall_risk_level", ""),
            json.dumps(diseases, ensure_ascii=False),
            json.dumps(features or {}, ensure_ascii=False),
            json.dumps(recommendations, ensure_ascii=False),
            report.get("recording_quality", ""),
            ai_insight,
            audio_path,
            now
        ))

        # 保存趋势数据
        for d in diseases:
            trend_id = f"trend_{uuid.uuid4().hex[:8]}"
            conn.execute("""
                INSERT INTO health_trends (id, user_id, metric_name, metric_value, recorded_at)
                VALUES (?, ?, ?, ?, ?)
            """, (trend_id, user_id, d["disease"], d["risk_score"], now))

        conn.commit()
        conn.close()
        return record_id

    def get_user_records(self, user_id: str, limit: int = 20) -> List[dict]:
        """获取用户的健康记录"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM health_records WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        conn.close()

        records = []
        for row in rows:
            r = dict(row)
            r["diseases"] = json.loads(r["diseases_json"])
            r["features"] = json.loads(r["features_json"])
            r["recommendations"] = json.loads(r["recommendations_json"])
            del r["diseases_json"], r["features_json"], r["recommendations_json"]
            records.append(r)
        return records

    def get_trends(self, user_id: str, days: int = 30) -> Dict[str, List[dict]]:
        """获取健康趋势数据"""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM health_trends WHERE user_id = ? AND recorded_at > ? ORDER BY recorded_at",
            (user_id, since)
        ).fetchall()
        conn.close()

        trends = {}
        for row in rows:
            name = row["metric_name"]
            if name not in trends:
                trends[name] = []
            trends[name].append({"value": row["metric_value"], "date": row["recorded_at"][:10]})
        return trends

    def get_stats(self, user_id: str) -> dict:
        """获取用户统计数据"""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) as c FROM health_records WHERE user_id = ?", (user_id,)).fetchone()["c"]
        avg_score = conn.execute("SELECT AVG(overall_score) as avg FROM health_records WHERE user_id = ?", (user_id,)).fetchone()["avg"] or 0
        last = conn.execute("SELECT created_at FROM health_records WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,)).fetchone()
        conn.close()
        return {
            "total_checks": total,
            "avg_score": round(avg_score, 1),
            "last_check": last["created_at"] if last else None,
            "member_since": self.get_user(user_id)["created_at"] if self.get_user(user_id) else None,
        }


db = Database()
