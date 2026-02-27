# DeepGuard Pro - Fixes Summary

## ✅ Implemented Fixes

### 1. Chatbot - Gemini Vision Integration & Visual Evidence
- **Backend** (`api.py`): Enhanced `/chat/message` to:
  - Fetch job context with frame analyses from Gemini Vision
  - Build evidence list with URLs for annotated frames
  - Convert file paths to `/evidence/{job_id}/frame_xxx.jpg` URLs
  - Include Vision agent heatmap evidence
- **Frontend** (`ChatAssistant.tsx`): Uses `sendChatMessage` when on Results page (job_id from URL)
- **Static files**: Mount `/evidence` for serving annotated frames from `/app/data/evidence`

### 2. INCONCLUSIVE Results Fix
- **neurosymbolic.py**: 
  - Added confidence calculation from agent scores (was always 0)
  - Normalize agent confidence (0-100 → 0-1)
  - Return `confidence` in result
- **api.py**: Fallback to extract verdict/confidence from result JSON when File node missing
- **celery_tasks.py**: Redis publish for WebSocket status updates (jobs now broadcast progress)

### 3. Email System
- **email_notifier.py**: 
  - Gmail support: `MAIL_USE_TLS=true`, port 587
  - Optional `password_hint` in welcome email (set `SEND_PASSWORD_IN_WELCOME=true`)
- **api.py**: Login with email OR username; `send_welcome_email` with optional password

### 4. Image Upload
- **api.py**: Extended allowed extensions (.bmp, .tiff, .mp3, .wav, .flac, .ogg)
- **Upload.tsx**: Accept attribute for images/video/audio
- Images already routed to Vision agent via `select_agents_by_filetype`

### 5. Non-Async UI
- **WebSocketContext.tsx**: Background polling when `jobId` + `isProcessing`; persists across navigation
- **AnalysisBanner.tsx**: Persistent banner "Analiză în curs" / "Analiză completă! Vezi rezultate"
- **MainLayout.tsx**: Renders AnalysisBanner

### 6. Neo4j setup_knowledge.py
- Uses env vars `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (matches docker-compose)

### 7. Docker
- Added `EVIDENCE_DIR=/app/data/evidence` to coordinator env

---

## 📋 Verification Commands

```powershell
# Rebuild and start
docker-compose down
docker-compose build --no-cache coordinator
docker-compose up -d

# Init Neo4j (use correct password from .env)
docker exec deepguard_brain python setup_knowledge.py

# Test image upload
curl -X POST http://localhost:8000/chat_analyze -H "Authorization: Bearer <TOKEN>" -F "file=@test.jpg"

# Test chat (on Results page with job_id)
# POST /chat/message with { "job_id": "<id>", "message": "de ce acest video este inconclusive?" }
```

---

## 🔧 Environment Variables for Email (Gmail)

```env
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=deepguardpro@gmail.com
MAIL_PASSWORD=<app-password>
SEND_PASSWORD_IN_WELCOME=true  # Optional, for demo
```
