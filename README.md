# VoiceHealth

> Voice Biomarker AI Platform -- 30s Voice, 25 Disease Early Screening

[![Python](https://img.shields.io/badge/python-3.9+-green.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)]()
[![Version](https://img.shields.io/badge/v0.3-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-orange.svg)]()

## What It Does

**VoiceHealth sells health screening results, not voice analysis tools.** Speak for 30 seconds -> AI analyzes 59 acoustic features -> outputs risk assessment for 25 diseases.

> Core concept: Voice carries rich health information -- speech rate, pitch, tremor, pauses. These subtle changes appear before disease symptoms. VoiceHealth gives everyone a "voice checkup".

---

## Problem We Solve

| Pain Point | Traditional Checkup | VoiceHealth |
|-----------|-------------------|-----------|
| Frequency | 1-2x per year | **Anytime** |
| Invasiveness | Blood draw/Imaging | **Non-invasive** |
| Latency | Days for results | **Seconds** |
| Cost | Hundreds to thousands | **Free** |
| Coverage | Single indicator | **25 diseases at once** |

---

## System Architecture

```
+-----------------------------------------------------------+
|                  VoiceHealth v0.3                            |
+-----------------------------------------------------------+
|  User Layer                                                |
|  [Web H5]  [WeChat Mini]  [API Integration]               |
|       +-----------+-----------+                            |
|                   v                                        |
|  API Layer (FastAPI)                                      |
|  /api/v1/analyze   -- Voice detection + AI insight         |
|  /api/v1/records   -- History query                       |
|  /api/v1/trends    -- Health trend analysis               |
|  /api/v1/user      -- Register/Login/Profile              |
|  /api/v1/stats     -- User statistics                     |
|                   v                                        |
|  AI Engine Layer                                           |
|  [Acoustic Features]  [Disease Detection]  [MIMO AI]      |
|  [59-dim vector]      [25 diseases]        [Health Insight]|
|                   v                                        |
|  Data Layer (SQLite)                                      |
|  [User Profiles]  [Health Records]  [Trend Data]          |
+-----------------------------------------------------------+
```

---

## Acoustic Feature Engine

59-dimensional acoustic feature vector covering the full spectrum of voice biomarkers:

| Feature Category | Dimensions | Key Indicators |
|-----------------|-----------|---------------|
| MFCC | 26 | Spectral envelope |
| Fundamental Frequency (F0) | 6 | Pitch, variation, range |
| Jitter/Shimmer | 6 | Vocal fold vibration stability |
| HNR | 2 | Harmonic-to-noise ratio |
| Spectral Features | 6 | Centroid, bandwidth, flatness |
| Prosodic Features | 9 | Speech rate, pauses, rhythm |
| Formants | 4 | F1-F4, vocal tract shape |
| Energy Features | 2 | RMS energy and variation |

---

## Detection Coverage (25 Diseases)

| Category | Diseases |
|----------|---------|
| Mental Health (5) | Depression, Anxiety, Burnout, Stress, Social Isolation |
| Cognitive Decline (4) | Parkinson's, Alzheimer's, MCI, Frailty |
| Respiratory (3) | COPD, Asthma, Respiratory Distress |
| Cardiovascular (2) | Hypertension, Heart Failure |
| Metabolic (2) | Type 2 Diabetes, Thyroid Disorders |
| Health States (9) | Sleep, Fatigue, Alcohol, Pain, Emotion, Voice, Hearing, Cognitive Load, Autism Spectrum |

---

## AI Health Insights

Integrated with Xiaomi MIMO API, generating personalized health insights for each detection:
- Warm tone, non-alarming
- Specific actionable advice
- Not a medical diagnosis, always reminds for reference only

---

## Quick Start

```bash
git clone https://github.com/MoKangMedical/voicehealth.git
cd voicehealth
pip install -r requirements.txt
python3 -m src.api.main
# Visit http://localhost:8100
```

### Docker Deployment

```bash
docker-compose up -d
# Visit http://localhost:8100
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| VOICEHEALTH_PORT | 8100 | Server port |
| VOICEHEALTH_HOST | 0.0.0.0 | Server host |
| MIMO_API_KEY | (empty) | Xiaomi MIMO API key |
| MIMO_BASE_URL | https://api.xiaomimimo.com/v1 | MIMO API base URL |
| VOICEHEALTH_DB | data/voicehealth.db | SQLite database path |

---

## Harness Theory

VoiceHealth's core competitive advantage is the **Health Detection Harness** design:

```
Health Harness = Acoustic Feature Extraction + Disease Risk Models + AI Insight Generation + UX Design
```

- Same voice data, different Harness -> significant detection accuracy differences
- Our Harness encodes best practices from acoustic medical research
- Model weights can be open-sourced, Harness design is proprietary

---

## Tech Stack

- **Backend**: Python 3.9+, FastAPI, SQLite
- **Frontend**: Vanilla JS, Canvas API, Web Audio API
- **AI**: Xiaomi MIMO API (health insight generation)
- **Audio**: librosa, soundfile, numpy
- **Deployment**: Docker, uvicorn

---

## License

MIT License
