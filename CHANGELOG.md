# CHANGELOG

## v0.3.0 (2026-04-26)

### Frontend Overhaul
- Premium dark theme (Linear/Vercel-inspired design)
- Complete SVG icon system (zero emoji dependency)
- Canvas-based waveform visualization during recording
- Canvas-based trend charts in history and profile
- Animated score circle with arc drawing
- Smooth page transitions (fade + translateY)
- Filterable disease risk cards by category
- Mobile-first responsive layout (480px max-width)

### Backend Improvements
- Added uvicorn runner in main.py with env var configuration
- API key moved to environment variables (MIMO_API_KEY)
- Fallback insight generator when MIMO API unavailable
- Trend summary generation
- Added httpx dependency for async HTTP

### Infrastructure
- Added .gitignore (Python, DB, IDE, OS, audio files)
- Added Dockerfile for containerized deployment
- Added docker-compose.yml with volume persistence
- Added MIT License

## v0.2.0 (2026-04-22)

### C-end Complete Version
- User system: register/login/profile via phone number
- Health records: full CRUD with SQLite persistence
- Trend analysis: 30-day health trend tracking
- User statistics: total checks, average score, history
- AI insight: MIMO API integration for personalized health advice
- Disease info endpoint: categorized disease registry

## v0.1.0 (2026-04-22)

### Core Features
- Acoustic feature extraction engine: 59-dim feature vector (librosa + numpy)
- Disease detection engine: 25 diseases/conditions, 5 categories
- FastAPI backend: 3 API endpoints (/api/analyze, /api/diseases, /api/health)
- Web frontend: browser recording + real-time waveform + health report
- Offline demo mode: works without backend

### Detection Coverage
- Mental Health: depression, anxiety, burnout, stress, social isolation
- Cognitive Decline: Parkinson's, Alzheimer's, MCI, frailty
- Respiratory: COPD, asthma, respiratory distress
- Cardiovascular: hypertension, heart failure
- Metabolic: type 2 diabetes, thyroid disorders
- Health States: sleep, fatigue, alcohol, pain, emotion, voice, hearing, cognitive load, autism spectrum

### Known Limitations
- Disease detection uses rule-based heuristics (Mock mode)
- parselmouth not integrated (requires separate install)
- No real clinical data validation
