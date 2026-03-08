import os
import sys
import logging
import json
import hashlib
import time
import uuid
import asyncio
import httpx
import jwt
import base64
from typing import List, Optional, Dict, Any, Set, cast
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neo4j import GraphDatabase
from fastapi.staticfiles import StaticFiles
import redis.asyncio as aioredis


# --- 1. PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 2. CELERY IMPORT ---
try:
    from celery_tasks import process_media_analysis, celery_app
    CELERY_ENABLED = True
except ImportError:
    CELERY_ENABLED = False
    logging.error("❌ Celery tasks not available - running in sync mode")

# --- 3. MCP PROTOCOL SETUP ---
try:
    from mcp_protocol import mcp_router, integrate_with_api  # type: ignore
except ImportError as e:
    mcp_router = None
    logging.error(f"❌ mcp_protocol missing or failed to import: {e}")

# --- 4. AUTHENTICATION SETUP ---
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# --- 5. IMPORTURI DINAMICE ---
try:
    import neurosymbolic  # type: ignore
    from neurosymbolic import NeuroSymbolicDebater  # type: ignore
except ImportError:
    NeuroSymbolicDebater = None

try:
    from email_notifier import EmailNotifier  # type: ignore
    email_tool = EmailNotifier()
except ImportError:
    email_tool = None

try:
    import s3_manager as s3_lib  # type: ignore
    s3_manager = s3_lib.S3Manager()
except ImportError:
    s3_lib = None
    s3_manager = None

try:
    from tools.synthid_sentinel import SynthIDSentinel  # type: ignore
    synthid_tool = SynthIDSentinel()
except ImportError:
    synthid_tool = None
    logging.warning("⚠️ SynthID Sentinel not found.")

# --- 6. VERTEX AI IMPORT ---
try:
    from vertex_ai_helper import vertex_ai_manager  # type: ignore
except ImportError:
    vertex_ai_manager = None
    logging.warning("⚠️ Vertex AI helper not available")

# --- 7. GEMINI IMPORT (kept for compatibility — chat & vision now use GPT-4o) ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
gemini_client = None
try:
    from google import genai as _genai  # type: ignore
    _api_key = os.getenv("GEMINI_API_KEY")
    if _api_key:
        gemini_client = _genai.Client(api_key=_api_key)  # type: ignore
        logging.info("✅ Gemini client initialized (google-genai SDK)")
    else:
        logging.info("ℹ️ GEMINI_API_KEY not set — Gemini disabled (GPT-4o handles chat & vision)")
except ImportError:
    logging.warning("⚠️ google-genai package not available — chat uses GPT-4o instead")
    gemini_client = None

# --- 7b. OPENAI CLIENT (GPT-4o Vision + Chat fallback) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ENABLE_GPT4_VISION = os.getenv("ENABLE_GPT4_VISION", "true").lower() == "true"
GPT4_VISION_MODEL = os.getenv("GPT4_VISION_MODEL", "gpt-4o")
openai_client = None
try:
    if OPENAI_API_KEY:
        from openai import OpenAI as _OpenAI  # type: ignore
        openai_client = _OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)
        logging.info(f"✅ OpenAI client initialized (model: {GPT4_VISION_MODEL})")
    else:
        logging.warning("⚠️ OPENAI_API_KEY not set — GPT-4o Vision disabled")
except ImportError:
    logging.error("❌ openai package missing. Install: pip install openai>=1.30.0")
    openai_client = None
except Exception as _oe:
    logging.error(f"❌ OpenAI client init failed: {_oe}")
    openai_client = None

# --- 8. APP INITIALIZATION ---
app = FastAPI(
    title="DeepGuard Pro Coordinator",
    description="Neural Multi-Agent Deepfake Detection System",
    version="2.0.0"
)

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=True
)

if mcp_router:
    app.include_router(mcp_router)

# ============================================================================
# JWT SECRET KEY (64+ bytes secure)
# ============================================================================
SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", 
    "deepguard_production_secure_jwt_secret_key_2025_minimum_64_bytes_required_for_sha256_hmac_algorithm_compliant"
)
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

AGENT_REGISTRY = {
    "vision": "http://deepguard_vision:8002/analyze",
    "audio": "http://deepguard_audio:8004/analyze",
    "video": "http://deepguard_video:8003/analyze",
    "bio": "http://deepguard_bio:8005/analyze",
    "text": "http://deepguard_text:8001/analyze"
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepGuard.Coordinator")

# ============================================================================
# STATIC FILES: Annotated evidence frames from Gemini Vision
# ============================================================================
EVIDENCE_DIR = os.getenv("EVIDENCE_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "evidence"))
EVIDENCE_DIR = os.path.abspath(EVIDENCE_DIR)
if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
try:
    app.mount("/evidence", StaticFiles(directory=EVIDENCE_DIR), name="evidence")
    logger.info(f"✅ Evidence static files mounted at /evidence -> {EVIDENCE_DIR}")
except Exception as e:
    logger.warning(f"⚠️ Evidence mount failed: {e}")

# ... (Neo4jConnection and Redis code remains same) ...

# ============================================================================
# FILE UPLOAD CONFIGURATION
# ============================================================================
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_VIDEO_TYPES = ["video/mp4", "video/avi", "video/mov", "video/mkv", "video/webm"]
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"]
ALLOWED_MIME_TYPES = ALLOWED_VIDEO_TYPES + ALLOWED_IMAGE_TYPES

# ... (Previous imports and setup) ...



# ============================================================================
# NEO4J CONNECTION (GLOBAL)
# ============================================================================
class Neo4jConnection:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "deepguard_secret")
        self.driver = None
        self._connect()

    def _connect(self):
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50
            )
            self.driver.verify_connectivity()
            logger.info("✅ Neo4j connected")
        except Exception as e:
            logger.error(f"❌ Neo4j connection error: {e}")
            self.driver = None

    def query(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # Auto-reconnect if driver is missing
        if not self.driver:
            logger.warning("⚠️ Neo4j driver not active, attempting reconnection...")
            self._connect()
            
        if not self.driver:
            logger.error("❌ Neo4j driver still not initialized after retry")
            return []
            
        try:
            with self.driver.session() as session:
                result = session.run(cast(Any, cypher), parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"❌ Neo4j query failed: {e}")
            # If we get a connectivity error, force reconnect next time
            if "ServiceUnavailable" in str(e) or "SessionExpired" in str(e):
                self.close()
            return []

    def close(self):
        if self.driver:
            try:
                self.driver.close()
                logger.info("✅ Neo4j connection closed")
            except:
                pass
            finally:
                self.driver = None

# Initialize Neo4j connection
neo4j_conn = Neo4jConnection()

# ============================================================================
# REDIS PUB/SUB for WebSocket Broadcasting
# ============================================================================
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
redis_client: Optional[aioredis.Redis] = None

async def get_redis_client() -> aioredis.Redis:
    """Get or create Redis client"""
    global redis_client
    if redis_client is None:
        redis_client = await aioredis.from_url(
            REDIS_URL, 
            encoding="utf-8", 
            decode_responses=True,
            max_connections=50
        )
    return redis_client

async def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
    logger.info("✅ Redis connection closed")

def _init_neo4j_schema():
    """Create Neo4j constraints and indexes if they don't exist."""
    try:
        neo4j_conn.query("CREATE CONSTRAINT user_username IF NOT EXISTS FOR (u:User) REQUIRE u.username IS UNIQUE")
        neo4j_conn.query("CREATE CONSTRAINT job_id IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE")
        neo4j_conn.query("CREATE CONSTRAINT file_hash IF NOT EXISTS FOR (f:File) REQUIRE f.hash IS UNIQUE")
        neo4j_conn.query("CREATE INDEX job_status IF NOT EXISTS FOR (j:Job) ON (j.status)")
        logger.info("✅ Neo4j schema initialized")
    except Exception as e:
        logger.warning(f"⚠️ Neo4j schema init warning (may already exist): {e}")

def _init_admin_user():
    """Create default admin user if it doesn't exist."""
    try:
        admin_password = os.getenv("ADMIN_PASSWORD", "deepguard123")
        admin_email = os.getenv("ADMIN_EMAIL", "admin@deepguard.pro")
        hashed = pwd_context.hash(admin_password)
        neo4j_conn.query("""
            MERGE (u:User {username: 'admin'})
            ON CREATE SET u.hashed_password = $hp, u.email = $email, u.role = 'admin', u.created_at = datetime()
            ON MATCH SET u.role = 'admin', u.hashed_password = $hp, u.email = $email
        """, {"hp": hashed, "email": admin_email})
        logger.info("✅ Admin user initialized")
    except Exception as e:
        logger.warning(f"⚠️ Admin user init warning: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    try:
        await get_redis_client()
        logger.info("✅ Redis client initialized")
    except Exception as e:
        logger.error(f"❌ Redis initialization failed: {e}")
    try:
        _init_neo4j_schema()
    except Exception as e:
        logger.error(f"❌ Neo4j schema init failed: {e}")
    try:
        _init_admin_user()
    except Exception as e:
        logger.error(f"❌ Admin user init failed: {e}")
    logger.info("🚀 DeepGuard Pro Coordinator started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await close_redis()
    neo4j_conn.close()
    logger.info("🛑 DeepGuard Pro Coordinator stopped")

# ============================================================================
# WEBSOCKET BROADCASTING
# ============================================================================
active_websockets: Set[WebSocket] = set()

async def broadcast_status(message: str, status: str = "info", job_id: Optional[str] = None, progress: int = 0):
    """
    Broadcast status update to all connected WebSocket clients
    AND publish to Redis for cross-instance communication
    """
    payload = {
        "type": "status", 
        "status": status, 
        "message": message, 
        "timestamp": datetime.now().isoformat(),
        "job_id": job_id,
        "progress": progress
    }
    
    # Broadcast to local WebSocket connections
    disconnected = []
    for ws in active_websockets:
        try: 
            await ws.send_json(payload)
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            disconnected.append(ws)
    
    # Remove disconnected clients
    for ws in disconnected:
        active_websockets.discard(ws)
    
    # Publish to Redis for other instances
    try:
        client = await get_redis_client()
        await client.publish("deepguard_status_updates", json.dumps(payload))
    except Exception as e:
        logger.error(f"Redis publish error: {e}")

# ============================================================================
# LEGACY SYNC PIPELINE (Fallback if Celery unavailable)
# ============================================================================
async def select_agents_mcp(file_type: str, query: Optional[str]) -> List[str]:
    """Legacy agent selection — rule-based (no LLM needed)"""
    ft = file_type.lower()
    if any(ext in ft for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]):
        return ["video", "bio", "audio"]
    elif any(ext in ft for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]):
        return ["vision"]
    elif any(ext in ft for ext in [".mp3", ".wav", ".ogg", ".flac", ".m4a"]):
        return ["audio"]
    elif any(ext in ft for ext in [".txt", ".pdf", ".doc", ".docx"]):
        return ["text"]
    else:
        return ["vision"]

async def call_agent(name: str, endpoint: str, content: bytes, filename: str) -> Dict:
    """Legacy direct agent call"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {'file': (filename, content)}
            res = await client.post(endpoint, files=files)
            return res.json()
    except Exception as e: 
        return {"agent": name, "verdict": "ERROR", "error": str(e)}

async def process_pipeline_legacy(job_id: str, username: str, f_hash: str, filename: str, content: bytes, file_type: str, query: Optional[str]):
    """LEGACY SYNC PIPELINE - Only used if Celery is not available"""
    try:
        await broadcast_status(f"🧠 Cortex analizează {filename}...", "thinking", job_id)
        selected = await select_agents_mcp(file_type, query)
        
        tasks = [call_agent(a, AGENT_REGISTRY[a], content, filename) for a in selected if a in AGENT_REGISTRY]
        
        # SynthID Check
        synthid_result = None
        if synthid_tool and hasattr(synthid_tool, 'enabled') and synthid_tool.enabled and any(x in file_type.lower() for x in ['image', 'video']):
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                await broadcast_status("🔍 Verificare Watermark Digital (Google SynthID)...", "thinking", job_id)
                synthid_result = await synthid_tool.verify_media(tmp_path)
                os.remove(tmp_path)
            except Exception as e:
                logger.error(f"SynthID Error: {e}")

        raw_results = await asyncio.gather(*tasks)
        agent_results = {selected[i]: raw_results[i] for i in range(len(raw_results))}
        logger.info(f"🔍 Agent Results for {job_id}: {json.dumps(agent_results)}")
        
        final_verdict = "INCONCLUSIVE"
        reasoning = "Logic core unavailable."

        # SynthID Override
        if synthid_result and synthid_result.get("detected"):
            final_verdict = "FAKE"
            reasoning = "🔴 Google SynthID a detectat un watermark digital invizibil."
            agent_results["synthid"] = synthid_result
            await broadcast_status("🚨 SynthID Alert: Watermark Detectat!", "warning", job_id)
        elif NeuroSymbolicDebater is not None:
            if synthid_result: 
                agent_results["synthid"] = synthid_result
            debater = NeuroSymbolicDebater()  # type: ignore
            logic_result = debater.validate_contradictions(agent_results)
            final_verdict = logic_result["final_verdict"]
            reasoning = logic_result["reasoning"]

        full_result = {
            "verdict": final_verdict,
            "reasoning": reasoning,
            "confidence": 0.0,
            "agents_used": selected,
            "details": agent_results
        }

        serialized_result = json.dumps(full_result)

        s3_url = ""
        if s3_manager and content:
            try:
                s3_url = s3_manager.upload_file(content, f"{f_hash}_{filename}")
            except Exception as s3_err:
                logger.warning(f"⚠️ S3 upload failed for {filename}: {s3_err}")

        neo4j_conn.query("""
            MATCH (u:User {username: $u})-[:REQUESTED]->(j:Job {id: $job_id})
            MERGE (f:File {hash: $hash})
            SET f.verdict = $verdict, f.reasoning = $reason, f.url = $url, f.timestamp = datetime()
            SET j.status = 'completed', j.result = $serialized_result, j.progress = 100
            MERGE (u)-[:OWNS]->(f)
            MERGE (j)-[:RESULT]->(f)
        """, {"job_id": job_id, "u": username, "hash": f_hash, "verdict": final_verdict, "reason": reasoning, "url": s3_url, "serialized_result": serialized_result})

        if email_tool:
            user_data = neo4j_conn.query("MATCH (u:User {username: $u}) RETURN u.email as email", {"u": username})
            if user_data and user_data[0].get('email'):
                email_tool.send_analysis_complete(str(user_data[0]['email']), filename, final_verdict, reasoning, job_id=job_id)

        await broadcast_status(f"✅ Gata! Verdict: {final_verdict}", "success", job_id)
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        neo4j_conn.query("MATCH (j:Job {id: $id}) SET j.status = 'failed', j.progress = 0", {"id": job_id})
        await broadcast_status(f"❌ Eroare: {str(e)}", "error", job_id)

# ============================================================================
# AUTHENTICATION
# ============================================================================
async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        results = neo4j_conn.query("MATCH (u:User {username: $u}) RETURN u", {"u": payload.get("sub")})
        if results and len(results) > 0:
            return cast(Dict[str, Any], results[0]['u'])
    except:
        pass
    raise HTTPException(status_code=401, detail="Invalid authentication credentials")

async def require_admin(current_user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Dependency that requires admin role."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# ============================================================================
# SIMPLE RATE LIMITER (no external dependency)
# ============================================================================
from collections import defaultdict
import time as _time

_rate_limit_store: Dict[str, list] = defaultdict(list)
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX = 10     # max requests per window

def check_rate_limit(client_ip: str, endpoint: str = "auth"):
    """Simple in-memory rate limiter. Returns True if request is allowed."""
    key = f"{client_ip}:{endpoint}"
    now = _time.time()
    # Clean old entries
    _rate_limit_store[key] = [t for t in _rate_limit_store[key] if now - t < _RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[key]) >= _RATE_LIMIT_MAX:
        return False
    _rate_limit_store[key].append(now)
    return True

# ============================================================================
# AUTH ROUTES
# ============================================================================
@app.post("/auth/register")
async def register(request: Request):
    data = await request.json()

    # Rate limiting check
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip, "register"):
        raise HTTPException(status_code=429, detail="Too many registration attempts. Try again later.")

    u = str(data.get("username", ""))
    p = str(data.get("password", ""))
    email = str(data.get("email", ""))
    
    if not u or not p:
        raise HTTPException(status_code=400, detail="Missing username or password")

    existing = neo4j_conn.query("MATCH (u:User {username: $u}) RETURN u", {"u": u})
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed = pwd_context.hash(p)
    neo4j_conn.query("""
        CREATE (u:User {username: $u, hashed_password: $p, email: $e, created_at: datetime(), role: 'user'})
    """, {"u": u, "p": hashed, "e": email})
    
    token = jwt.encode({"sub": u, "exp": datetime.now() + timedelta(hours=24)}, SECRET_KEY, algorithm=ALGORITHM)
    
    # Send welcome email on registration
    if email and email_tool:
        try:
            email_tool.send_welcome_email(email, u)
        except Exception as e:
            logger.error(f"Welcome email error: {e}")

    return {"access_token": token, "token_type": "bearer"}

@app.post("/auth/login")
async def login(request: Request):
    data = await request.json()

    # Rate limiting check
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip, "login"):
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.")

    u, p = str(data.get("username", data.get("email", ""))), str(data.get("password", ""))
    
    # Support login with username OR email
    res = neo4j_conn.query("""
        MATCH (u:User) 
        WHERE u.username = $u OR u.email = $u
        RETURN u.username as username, u.hashed_password as hp, u.email as email
    """, {"u": u})
    
    if res and pwd_context.verify(p, str(res[0]['hp'])):
        token = jwt.encode({"sub": res[0].get('username', u), "exp": datetime.now() + timedelta(hours=24)}, SECRET_KEY, algorithm=ALGORITHM)
        logged_username = res[0].get('username', u)
        # Send login alert email
        if email_tool:
            login_email = (res[0].get('email') or '').strip()
            if login_email:
                try:
                    email_tool.send_login_alert(login_email, logged_username)
                except Exception as e:
                    logger.warning(f"Login alert email failed: {e}")
        return {"access_token": token, "token_type": "bearer"}

    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/auth/me")
async def read_users_me(current_user: Dict = Depends(get_current_user)):
    return {
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "disabled": False,
        "role": current_user.get("role", "user")
    }

# ============================================================================
# MAIN ANALYSIS ENDPOINT
# ============================================================================
@app.post("/chat_analyze")
async def chat_analyze(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    query: Optional[str] = Form(None),
    force_refresh: Optional[str] = Form(None),
    current_user: Dict = Depends(get_current_user)
):
    import mimetypes
    from pathlib import Path

    job_id = str(uuid.uuid4())
    username = str(current_user.get('username', 'anonymous'))
    
    # 1. Handle File Upload
    if file:
        # Validate MIME type (if available) behavior checks
        if file.content_type not in ALLOWED_MIME_TYPES:
             # Fallback: check extension if content-type is generic
             mime_type, _ = mimetypes.guess_type(file.filename or "")
             if not mime_type or mime_type not in ALLOWED_MIME_TYPES:
                 # Last chance: trusting the extension for now, but logging warning
                 logger.warning(f"⚠️ Suspicious MIME type: {file.content_type} for {file.filename}")
                 # For now, we allow it if it looks like an image/video extension, strictly for robustness
                 ext = os.path.splitext(file.filename or "")[1].lower()
                 if ext not in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.mp3', '.wav', '.flac', '.ogg']:
                     raise HTTPException(
                        status_code=400, 
                        detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(ALLOWED_MIME_TYPES)}"
                     )

        content = await file.read()
        file_size = len(content)
        
        if file_size > MAX_FILE_SIZE:
             raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE} bytes)"
            )
            
        f_hash = hashlib.sha256(content).hexdigest()
        safe_filename = str(file.filename)
        
        # Detect MIME type properly
        mime_type = file.content_type
        if not mime_type or mime_type == "application/octet-stream":
            mime_type, _ = mimetypes.guess_type(safe_filename)
            if not mime_type:
                mime_type = "application/octet-stream"
                
        file_type = os.path.splitext(safe_filename)[1] or ".bin"

        # Extract resolution for images and videos
        resolution = None
        try:
            if mime_type and mime_type.startswith("image/"):
                import io as _io
                from PIL import Image as _PilImage
                img = _PilImage.open(_io.BytesIO(content))
                resolution = f"{img.width}x{img.height}"
            elif mime_type and mime_type.startswith("video/"):
                import tempfile as _tmp, subprocess as _sp
                with _tmp.NamedTemporaryFile(delete=False, suffix=file_type) as _tf:
                    _tf.write(content)
                    _tf_path = _tf.name
                try:
                    _r = _sp.run(
                        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                         "-show_entries", "stream=width,height", "-of", "csv=p=0", _tf_path],
                        capture_output=True, text=True, timeout=10
                    )
                    _parts = _r.stdout.strip().split(",")
                    if len(_parts) == 2 and _parts[0].isdigit():
                        resolution = f"{_parts[0]}x{_parts[1]}"
                finally:
                    try: os.unlink(_tf_path)
                    except: pass
        except Exception as _e:
            logger.debug(f"Could not extract resolution: {_e}")

    else:
        # Text-only query
        content = b""
        f_hash = f"chat_{int(time.time())}"
        safe_filename = "query_input"
        file_type = ".txt"
        file_size = 0
        mime_type = "text/plain"
        resolution = None

    # 2. Deduplication — check if this file was already analyzed (skip if force_refresh)
    skip_cache = force_refresh and force_refresh.lower() in ("true", "1", "yes")
    if content and f_hash and not f_hash.startswith("chat_") and not skip_cache:
        cached = neo4j_conn.query("""
            MATCH (f:File {hash: $hash})
            WHERE f.verdict IS NOT NULL
            RETURN f.verdict as verdict, f.confidence as confidence, f.reasoning as reasoning, f.timestamp as timestamp
        """, {"hash": f_hash})

        if cached and cached[0].get("verdict"):
            logger.info(f"🔁 Cache hit for hash {f_hash[:16]}... — returning previous result")
            # Create a new job that instantly completes with cached result
            cached_job_id = str(uuid.uuid4())
            neo4j_conn.query("""
                MATCH (u:User {username: $u})
                CREATE (j:Job {
                    id: $id, status: 'completed', progress: 100,
                    created_at: datetime(), completed_at: datetime(),
                    file_hash: $hash, filename: $filename, cached: true
                })
                CREATE (u)-[:REQUESTED]->(j)
                WITH j
                MATCH (f:File {hash: $hash})
                MERGE (j)-[:RESULT]->(f)
            """, {"u": username, "id": cached_job_id, "hash": f_hash, "filename": safe_filename})

            return {
                "job_id": cached_job_id,
                "status": "completed",
                "cached": True,
                "verdict": cached[0].get("verdict"),
                "confidence": cached[0].get("confidence"),
                "message": "Fișierul a fost deja analizat — rezultat din cache."
            }

    # 3. Create Job in Neo4j
    neo4j_conn.query("""
        MATCH (u:User {username: $u})
        CREATE (j:Job {
            id: $id,
            status: 'queued',
            progress: 0,
            created_at: datetime(),
            file_hash: $hash,
            filename: $filename,
            file_size: $file_size,
            mime_type: $mime_type,
            resolution: $resolution
        })
        CREATE (u)-[:REQUESTED]->(j)
    """, {
        "u": username,
        "id": job_id,
        "hash": f_hash,
        "filename": safe_filename,
        "file_size": file_size,
        "mime_type": mime_type,
        "resolution": resolution
    })

    # 3. Dispatch Task
    if CELERY_ENABLED:
        content_b64 = base64.b64encode(content).decode('utf-8')
        
        task = process_media_analysis.delay(
            job_id=job_id,
            username=username,
            f_hash=f_hash,
            filename=safe_filename,
            content_b64=content_b64,
            file_type=file_type,
            query=query
        )
        
        logger.info(f"🚀 Celery task dispatched: {task.id} for job {job_id}")
        
        neo4j_conn.query("""
            MATCH (j:Job {id: $job_id})
            SET j.celery_task_id = $task_id, j.status = 'queued'
        """, {"job_id": job_id, "task_id": task.id})
        
    else:
        logger.warning(f"⚠️ Celery not available, using BackgroundTasks for job {job_id}")
        background_tasks.add_task(
            process_pipeline_legacy, 
            job_id, username, f_hash, safe_filename, content, file_type, query
        )
    
    return {
        "job_id": job_id, 
        "status": "queued",
        "message": "Analiza a fost trimisă spre procesare."
    }

# ============================================================================
# JOB STATUS & HISTORY
# ============================================================================
@app.get("/jobs/my-active")
async def get_my_active_jobs_early(current_user: Dict = Depends(get_current_user)):
    """Returnează job-urile active (queued/processing) ale userului curent."""
    username = current_user.get("username")
    results = neo4j_conn.query("""
        MATCH (u:User {username: $username})-[:REQUESTED]->(j:Job)
        WHERE j.status IN ['queued', 'processing']
        RETURN j.id as job_id, j.status as status, j.progress as progress,
               j.status_message as status_message, j.filename as filename,
               j.created_at as created_at
        ORDER BY j.created_at DESC
        LIMIT 5
    """, {"username": username})
    return {"active_jobs": results or []}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, current_user: Dict = Depends(get_current_user)):
    username = current_user.get("username", "")
    result = neo4j_conn.query("""
        MATCH (j:Job {id: $job_id})<-[:REQUESTED]-(u:User {username: $username})
        OPTIONAL MATCH (j)-[:RESULT]->(f:File)
        RETURN j.status as status,
               j.progress as progress,
               j.status_message as message,
               j.result as result,
               j.celery_task_id as task_id,
               f.verdict as final_verdict,
               f.reasoning as reasoning,
               f.confidence as confidence,
               j.created_at as created_at,
               j.completed_at as completed_at,
               j.filename as filename,
               j.file_size as file_size,
               j.mime_type as mime_type,
               j.resolution as resolution,
               j.osint_results as osint_results,
               j.osint_risk_level as osint_risk_level,
               j.osint_completed_at as osint_completed_at
    """, {"job_id": job_id, "username": username})
    
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")
    
    data = result[0]
    
    if data.get('result') and isinstance(data['result'], str):
        try:
            data['result'] = json.loads(data['result'])
        except:
            pass
    
    # Fallback: extract verdict/confidence from result JSON if File node missing
    if not data.get('final_verdict') and data.get('result') and isinstance(data['result'], dict):
        res = data['result']
        data['final_verdict'] = res.get('verdict')
        data['reasoning'] = data.get('reasoning') or res.get('reasoning')
        data['confidence'] = data.get('confidence') if data.get('confidence') is not None else res.get('confidence')
    
    if CELERY_ENABLED and data.get('task_id') and data.get('status') in ['queued', 'processing']:
        try:
            task_result = celery_app.AsyncResult(data['task_id'])
            data['celery_state'] = task_result.state
            if task_result.state == 'FAILURE':
                data['error'] = str(task_result.info)
        except Exception as e:
            logger.error(f"Celery status check error: {e}")

    # Parse OSINT results if stored as string
    osint_results = data.get("osint_results")
    if isinstance(osint_results, str):
        try:
            osint_results = json.loads(osint_results)
        except Exception:
            pass

    # Gracefully handle return structure
    return {
        "job_id": job_id,
        "status": data.get("status", "unknown"),
        "progress": data.get("progress", 0),
        "message": data.get("message", ""),
        "verdict": data.get("final_verdict"),
        "confidence": data.get("confidence"),
        "reasoning": data.get("reasoning"),
        "filename": data.get("filename", "Unknown"),
        "file_size": data.get("file_size"),
        "mime_type": data.get("mime_type"),
        "resolution": data.get("resolution"),
        "created_at": data.get("created_at"),
        "completed_at": data.get("completed_at"),
        "result": data.get("result"),
        "celery_state": data.get("celery_state"),
        "osint_results": osint_results,
        "osint_risk_level": data.get("osint_risk_level"),
        "osint_completed_at": data.get("osint_completed_at")
    }

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, current_user: Dict = Depends(get_current_user)):
    """Cancel a running analysis job"""
    username = current_user.get("username")

    # Verify job belongs to user and is still running
    result = neo4j_conn.query("""
        MATCH (u:User {username: $u})-[:REQUESTED]->(j:Job {id: $job_id})
        RETURN j.status as status, j.celery_task_id as celery_task_id
    """, {"u": username, "job_id": job_id})

    if not result:
        raise HTTPException(status_code=404, detail="Job not found")

    status = result[0].get("status", "")
    if status in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Job already {status}")

    # Revoke Celery task if available
    celery_task_id = result[0].get("celery_task_id")
    if celery_task_id:
        try:
            from celery_tasks import celery_app  # type: ignore
            celery_app.control.revoke(celery_task_id, terminate=True)
            logger.info(f"🛑 Revoked Celery task {celery_task_id} for job {job_id}")
        except Exception as e:
            logger.warning(f"⚠️ Could not revoke Celery task: {e}")

    # Update job status
    neo4j_conn.query("""
        MATCH (j:Job {id: $job_id})
        SET j.status = 'cancelled', j.progress = 0, j.cancelled_at = datetime()
    """, {"job_id": job_id})

    return {"status": "cancelled", "job_id": job_id}

@app.get("/history")
async def get_history(limit: int = 50, offset: int = 0, current_user: Dict = Depends(get_current_user)):
    """Get analysis history with pagination"""
    username = current_user.get("username")

    # Clamp values
    limit = max(1, min(100, limit))
    offset = max(0, offset)

    # Get total count
    count_result = neo4j_conn.query("""
        MATCH (u:User {username: $u})-[:REQUESTED]->(j:Job)
        RETURN count(j) as total
    """, {"u": username})
    total = count_result[0].get("total", 0) if count_result else 0

    # Get paginated records
    records = neo4j_conn.query("""
        MATCH (u:User {username: $u})-[:REQUESTED]->(j:Job)
        OPTIONAL MATCH (j)-[:RESULT]->(f:File)
        RETURN j.id as job_id,
               j.file_hash as file_hash,
               j.filename as filename,
               j.status as status,
               j.progress as progress,
               f.verdict as verdict,
               f.confidence as confidence,
               f.timestamp as timestamp,
               f.url as url,
               j.created_at as created_at
        ORDER BY j.created_at DESC
        SKIP $offset
        LIMIT $limit
    """, {"u": username, "limit": limit, "offset": offset})

    return {"records": records, "total": total, "limit": limit, "offset": offset}

# ============================================================================
# CHATBOT ENDPOINTS
# ============================================================================
class ChatRequest(BaseModel):
    job_id: Optional[str] = None
    message: str
    context: Optional[str] = None

@app.post("/chat/message")
async def chat_message(req: ChatRequest, request: Request, current_user: Dict = Depends(get_current_user)):
    """
    Interactive Q&A about an analysis job.
    Uses Gemini to answer questions based on the evidence collected.
    Returns annotated frames as evidence with accessible URLs.
    """
    username = current_user.get('username')
    job_context = {}
    
    # 1. Fetch Job Context if job_id provided
    if req.job_id:
        try:
            result = neo4j_conn.query("""
                MATCH (j:Job {id: $job_id})<-[:REQUESTED]-(u:User {username: $u})
                OPTIONAL MATCH (j)-[:RESULT]->(f:File)
                RETURN j.result as result, f.verdict as verdict, f.reasoning as reasoning
            """, {"job_id": req.job_id, "u": username})
            
            if result and result[0].get('result'):
                raw = result[0]['result']
                job_context = raw if isinstance(raw, dict) else json.loads(raw) if isinstance(raw, str) else {}
        except Exception as e:
            logger.error(f"Failed to fetch job context: {e}")

    # 2. Build evidence list with URLs for annotated frames
    evidence = []
    base_url = str(request.base_url).rstrip("/")
    
    # From Gemini Vision annotated_frames (file paths -> URLs)
    annotated = job_context.get("annotated_frames", [])
    frame_analyses = job_context.get("details", {}).get("gemini_vision", {}).get("frame_analyses", [])
    
    for i, frame_path in enumerate(annotated):
        # Convert /app/data/evidence/{job_id}/frame_001_FAKE.jpg -> job_id/frame_001_FAKE.jpg
        if frame_path:
            if "evidence" in frame_path:
                parts = frame_path.split("evidence" + os.sep, 1)
                rel = parts[-1].replace("\\", "/") if len(parts) > 1 else os.path.basename(frame_path)
            elif os.path.exists(frame_path):
                rel = os.path.relpath(frame_path, EVIDENCE_DIR).replace("\\", "/")
            else:
                rel = os.path.basename(frame_path)
            frame_info = frame_analyses[i] if i < len(frame_analyses) else {}
            evidence.append({
                "type": "frame",
                "title": f"Frame {frame_info.get('extracted_index', i+1)} la {frame_info.get('timestamp', 0):.1f}s",
                "url": f"{base_url}evidence/{rel}",
                "description": frame_info.get("reasoning", "Frame analizat de Gemini Vision"),
                "verdict": frame_info.get("verdict", "INCONCLUSIVE"),
                "confidence": (frame_info.get("confidence", 0) * 100) if isinstance(frame_info.get("confidence"), (int, float)) and frame_info.get("confidence", 0) <= 1 else frame_info.get("confidence", 0),
                "artifacts": frame_info.get("artifacts_detected", [])
            })
    
    # From Vision agent heatmap
    vision_details = job_context.get("details", {}).get("vision", {})
    if isinstance(vision_details, dict) and vision_details.get("evidence"):
        for ev in vision_details["evidence"]:
            if ev.get("url"):
                evidence.append({
                    "type": ev.get("type", "heatmap"),
                    "title": ev.get("title", "Saliency Map"),
                    "url": f"{base_url}{ev['url'].lstrip('/')}" if not ev["url"].startswith("http") else ev["url"],
                    "description": ev.get("description", ""),
                    "confidence": ev.get("confidence")
                })

    # 3. Construct System Prompt with frame references
    frame_refs = ""
    if frame_analyses:
        frame_refs = "\n\nFRAME-BY-FRAME ANALYSIS:\n"
        for fa in frame_analyses:
            frame_refs += f"- Frame {fa.get('extracted_index', '?')} ({fa.get('timestamp', 0):.1f}s): {fa.get('verdict', 'N/A')} - {fa.get('reasoning', '')[:100]}\n"

    system_prompt = f"""You are **Cortex**, the forensic AI assistant of DeepGuard Pro — a world-class deepfake detection platform.

## YOUR IDENTITY
- Expert forensic analyst with deep knowledge of GAN artifacts, face-swap techniques, diffusion model patterns
- Direct, precise, evidence-based communication style
- Use Romanian language for responses (same language as the user query if in Romanian)
- Use **bold** for key terms, - bullet points for lists, ## headers for sections when structuring long answers

## CURRENT ANALYSIS CONTEXT
- **Job ID**: {req.job_id or 'N/A'}
- **User**: {username}
- **Final Verdict**: {job_context.get('verdict', 'UNKNOWN')}
- **Confidence**: {job_context.get('confidence', 0):.1%}
- **Overall Reasoning**: {str(job_context.get('reasoning', 'N/A'))[:300]}
{frame_refs}

## AGENT RESULTS SUMMARY
{json.dumps({k: v for k, v in job_context.get('details', {}).items() if k in ['vision', 'audio', 'video', 'bio', 'text', 'gemini_vision']}, indent=2, default=str)[:2000]}

## USER QUESTION
{req.message}

## RESPONSE RULES
1. Answer DIRECTLY — no filler phrases like "Great question!" or "Certainly!"
2. Cite SPECIFIC evidence: frame numbers, timestamps, agent names, artifact names
3. If asked about a specific frame, reference its timestamp and detected artifacts
4. For "why FAKE" questions: list the TOP 3 most damning artifacts found
5. For "why REAL" questions: list the strongest authenticity indicators
6. Keep responses concise (max 300 words) unless a detailed breakdown is requested
7. End with a confidence qualifier if verdict confidence < 70%
8. NEVER make up evidence — only cite what is in the context above"""

    # 4. Call GPT-4o (OpenAI) — replaced Gemini
    try:
        if openai_client is None:
            raise RuntimeError("OpenAI client not available — check OPENAI_API_KEY")

        gpt_response = openai_client.chat.completions.create(
            model=GPT4_VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Cortex, the forensic AI assistant of DeepGuard Pro — "
                        "a world-class deepfake detection platform. "
                        "Always respond in Romanian. Be direct, evidence-based, and concise. "
                        "Never invent evidence — only cite what is provided in the context."
                    )
                },
                {"role": "user", "content": system_prompt}
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        reply = gpt_response.choices[0].message.content or "Răspuns gol de la GPT-4o."

        return {
            "role": "assistant",
            "content": reply,
            "evidence": evidence
        }

    except Exception as e:
        logger.error(f"Chat error (GPT-4o): {e}")
        return {
            "role": "assistant",
            "content": "Cortex core este indisponibil momentan. Verifică OPENAI_API_KEY.",
            "evidence": evidence,
            "error": str(e)
        }

@app.get("/chat/history")
async def get_chat_history(current_user: Dict = Depends(get_current_user)):
    # Placeholder: In a real app, fetch from Neo4j
    return {"messages": []}

# ============================================================================
# SYSTEM CONFIG
# ============================================================================
@app.post("/system/config/email")
async def update_email_config(request: Request, current_user: Dict = Depends(get_current_user)):
    data = await request.json()
    neo4j_conn.query("""
        MERGE (c:SystemConfig {id: 'main'})
        SET c.email_settings = $settings, c.updated_at = datetime()
    """, {"settings": json.dumps(data)})
    return {"status": "updated", "message": "Email configuration saved"}

# ============================================================================
# WEBSOCKET CHAT ENDPOINT (Chatbot Q&A)
# ============================================================================
def _build_chat_context(job: dict) -> dict:
    """Extrage contextul complet al analizei din datele Neo4j pentru chatbot."""
    filename = job.get("filename", "Unknown")
    verdict = job.get("verdict", "N/A")
    reasoning = job.get("reasoning", "N/A")
    confidence = job.get("confidence", 0) or 0
    if confidence > 1:
        confidence = confidence / 100

    details: dict = {}
    agents_used: list = []

    raw_result = job.get("result")
    if raw_result:
        try:
            parsed = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
            if not verdict or verdict == "N/A":
                verdict = parsed.get("verdict", verdict)
                reasoning = parsed.get("reasoning", reasoning)
                confidence = parsed.get("confidence", confidence) or confidence
            details = parsed.get("details", {})
            agents_used = parsed.get("agents_used", list(details.keys()))
        except Exception:
            pass

    # --- Rezumat per agent ---
    agent_lines = []
    for agent_name, agent_data in details.items():
        if not isinstance(agent_data, dict):
            continue
        av = str(agent_data.get("verdict", agent_data.get("is_ai_generated", "N/A"))).upper()
        fp = agent_data.get("fake_probability", agent_data.get("confidence", None))
        conf_str = f"{fp:.1f}%" if isinstance(fp, (int, float)) else "N/A"
        agent_lines.append(f"  • {agent_name}: {av} ({conf_str})")
    agent_summary = "\n".join(agent_lines) if agent_lines else "  N/A"

    # --- Fingerprint ---
    fp_data = details.get("fingerprint", {})
    detected_model = fp_data.get("detected_model") or fp_data.get("model_type")
    fp_conf = fp_data.get("confidence")
    if detected_model:
        fp_conf_str = f"{fp_conf*100:.0f}%" if isinstance(fp_conf, float) and fp_conf <= 1 else (f"{fp_conf:.0f}%" if fp_conf else "")
        fingerprint_info = f"Detectat: {detected_model} {fp_conf_str}".strip()
    else:
        fingerprint_info = "N/A"

    # --- Grad-CAM ---
    vision_data = details.get("vision", {})
    evidence_list = vision_data.get("evidence", [])
    gradcam_items = [e for e in evidence_list if isinstance(e, dict) and e.get("type") == "grad_cam"]
    gradcam_info = f"Disponibil la {gradcam_items[0].get('url', '')}" if gradcam_items else "N/A"

    # --- Frame analyses (Gemini) ---
    gemini_data = details.get("gemini_vision", {})
    frame_analyses = gemini_data.get("frame_analyses", [])
    frame_lines = []
    for fa in frame_analyses[:3]:
        ts = fa.get("timestamp", "?")
        fv = str(fa.get("verdict", "?")).upper()
        arts = fa.get("artifacts_detected", [])
        arts_str = ", ".join(arts[:3]) if arts else ""
        frame_lines.append(f"  • Frame la {ts}s: {fv}" + (f" — {arts_str}" if arts_str else ""))
    frames_info = "\n".join(frame_lines) if frame_lines else "N/A"

    # --- OSINT ---
    osint_raw = job.get("osint_results")
    osint_data: dict = {}
    if isinstance(osint_raw, str):
        try:
            osint_data = json.loads(osint_raw)
        except Exception:
            pass
    elif isinstance(osint_raw, dict):
        osint_data = osint_raw
    risk_level = job.get("osint_risk_level") or osint_data.get("risk_level")
    osint_summary_text = osint_data.get("summary", "")
    osint_info = f"Risc: {risk_level}. {osint_summary_text}".strip(" .") if risk_level else "N/A"

    return {
        "filename": filename,
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "agents_used": agents_used,
        "agent_summary": agent_summary,
        "fingerprint_info": fingerprint_info,
        "detected_model": detected_model,
        "gradcam_info": gradcam_info,
        "has_gradcam": bool(gradcam_items),
        "frames_info": frames_info,
        "has_frames": bool(frame_lines),
        "osint_info": osint_info,
        "has_osint": bool(risk_level),
        "mime_type": job.get("mime_type", ""),
    }


def _build_suggested_questions(ctx: dict) -> list:
    """Generează întrebări sugestive bazate pe contextul analizei."""
    verdict = ctx.get("verdict", "N/A")
    questions = [
        f"De ce e considerat {verdict}?",
        "Care agent a fost cel mai sigur în decizie?",
    ]
    if ctx.get("detected_model"):
        questions.append(f"Ce înseamnă că a fost creat cu {ctx['detected_model']}?")
    if ctx.get("has_gradcam"):
        questions.append("Explică heatmap-ul Grad-CAM — ce zone sunt suspecte?")
    if ctx.get("has_osint"):
        questions.append("Ce a găsit investigația OSINT?")
    if ctx.get("has_frames"):
        questions.append("Care frame din video a fost cel mai suspect?")
    if verdict in ("FAKE", "SUSPICIOUS"):
        questions.append("Ce ar putea face utilizatorul cu această informație?")
    return questions[:6]


@app.websocket("/ws/chat")
async def chat_websocket_endpoint(websocket: WebSocket):
    """WebSocket for interactive chat about analysis results. Query param: job_id"""
    await websocket.accept()
    job_id = websocket.query_params.get("job_id", "")
    conversation_history: list = []  # Memorie conversațională per sesiune

    try:
        if not job_id:
            await websocket.send_json({"type": "error", "message": "Job ID required"})
            await websocket.close()
            return

        # Fetch job o singură dată la conectare pentru suggested questions
        init_result = neo4j_conn.query("""
            MATCH (j:Job {id: $job_id})
            OPTIONAL MATCH (j)-[:RESULT]->(f:File)
            RETURN j.filename as filename, j.result as result, j.mime_type as mime_type,
                   j.osint_results as osint_results, j.osint_risk_level as osint_risk_level,
                   f.verdict as verdict, f.reasoning as reasoning, f.confidence as confidence
        """, {"job_id": job_id})

        if init_result:
            init_ctx = _build_chat_context(init_result[0])
            suggested = _build_suggested_questions(init_ctx)
            await websocket.send_json({
                "type": "suggested_questions",
                "questions": suggested
            })

        while True:
            data = await websocket.receive_json()
            user_question = data.get("message", "").strip()
            if not user_question:
                continue

            # Fetch job context (actualizat pentru fiecare mesaj — OSINT poate fi completat mai târziu)
            result = neo4j_conn.query("""
                MATCH (j:Job {id: $job_id})
                OPTIONAL MATCH (j)-[:RESULT]->(f:File)
                RETURN j.filename as filename, j.result as result, j.mime_type as mime_type,
                       j.osint_results as osint_results, j.osint_risk_level as osint_risk_level,
                       f.verdict as verdict, f.reasoning as reasoning, f.confidence as confidence
            """, {"job_id": job_id})

            if not result:
                await websocket.send_json({"type": "error", "message": "Job not found"})
                continue

            ctx = _build_chat_context(result[0])

            system_prompt = f"""Ești Cortex, asistentul AI forensic al platformei DeepGuard Pro.
Răspunde ÎNTOTDEAUNA în română. Fii direct, bazat pe dovezi și concis (max 3-4 propoziții).
Nu inventa informații — citează doar ce există în contextul de mai jos.

=== CONTEXT ANALIZĂ ===
Fișier: {ctx['filename']}
Verdict final: {ctx['verdict']} (confidență: {ctx['confidence']:.0%})
Agenți utilizați: {', '.join(ctx['agents_used']) if ctx['agents_used'] else 'N/A'}

=== VERDICTS PER AGENT ===
{ctx['agent_summary']}

=== FINGERPRINT (model generativ detectat) ===
{ctx['fingerprint_info']}

=== GRAD-CAM (explainability vizuală) ===
{ctx['gradcam_info']}

=== FRAME ANALYSIS (primele 3 frame-uri suspecte) ===
{ctx['frames_info']}

=== INVESTIGAȚIE OSINT ===
{ctx['osint_info']}

=== RAȚIONAMENT NEURO-SIMBOLIC ===
{ctx['reasoning'][:1500]}"""

            conversation_history.append({"role": "user", "content": user_question})

            # Limitează istoricul la ultimele 10 schimburi
            history_to_send = conversation_history[-10:]

            try:
                if openai_client:
                    ws_gpt_response = openai_client.chat.completions.create(
                        model=GPT4_VISION_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *history_to_send
                        ],
                        max_tokens=600,
                        temperature=0.3,
                    )
                    reply = ws_gpt_response.choices[0].message.content or "Răspuns gol."
                else:
                    # Fallback Gemini dacă OpenAI nu e disponibil
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
                        g_model = genai.GenerativeModel("gemini-2.0-flash")
                        g_resp = g_model.generate_content(
                            f"{system_prompt}\n\nÎntrebare utilizator: {user_question}"
                        )
                        reply = g_resp.text or "Răspuns gol."
                    except Exception:
                        reply = (
                            f"Analiza indică verdict **{ctx['verdict']}** cu confidență "
                            f"{ctx['confidence']:.0%}. {ctx['reasoning'][:300]}"
                        )
            except Exception as e:
                logger.error(f"Chat WebSocket AI error: {e}")
                reply = f"Eroare temporară la procesare. Încearcă din nou."

            conversation_history.append({"role": "assistant", "content": reply})
            await websocket.send_json({"type": "message", "content": reply})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Chat WebSocket error: {e}")

# ============================================================================
# WEBSOCKET ENDPOINT (FIXED)
# ============================================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time WebSocket connection with dedicated Redis pubsub per connection"""
    await websocket.accept()
    active_websockets.add(websocket)
    
    logger.info(f"✅ WebSocket connected. Total: {len(active_websockets)}")
    
    pubsub = None
    redis_task = None
    
    try:
        client = await get_redis_client()
        pubsub = client.pubsub()
        await pubsub.subscribe("deepguard_status_updates")
        
        async def redis_listener():
            try:
                async for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            await websocket.send_json(data)
                        except Exception as e:
                            logger.error(f"WebSocket message error: {e}")
                            continue
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Redis listener error: {e}")
        
        redis_task = asyncio.create_task(redis_listener())
        
        try:
            while True:
                data = await websocket.receive_text()
                if data:
                    await websocket.send_json({"type": "pong", "message": "Connection alive"})
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected normally")
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_websockets.discard(websocket)
        
        if redis_task:
            redis_task.cancel()
            try:
                await redis_task
            except asyncio.CancelledError:
                pass
        
        if pubsub:
            try:
                await pubsub.unsubscribe("deepguard_status_updates")
                await pubsub.close()
            except Exception as e:
                logger.error(f"Pubsub cleanup error: {e}")
        
        logger.info(f"✅ WebSocket cleaned up. Remaining: {len(active_websockets)}")

# ============================================================================
# PDF REPORT GENERATION
# ============================================================================
class ReportRequest(BaseModel):
    filename: str = "unknown"
    verdict: str = "INCONCLUSIVE"
    confidence: float = 0.0
    report_text: str = ""
    metadata: Optional[Dict[str, Any]] = None

@app.post("/generate_report")
async def generate_report(req: ReportRequest, current_user: Dict = Depends(get_current_user)):
    """Generate a professional PDF report for a completed analysis."""
    from fastapi.responses import StreamingResponse
    import io
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        raise HTTPException(status_code=500, detail="reportlab not installed")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=25*mm, bottomMargin=20*mm,
                            leftMargin=20*mm, rightMargin=20*mm)

    styles = getSampleStyleSheet()
    # Custom styles
    styles.add(ParagraphStyle(name='Title2', parent=styles['Title'],
                              fontSize=22, spaceAfter=4*mm,
                              textColor=HexColor('#1a1a2e')))
    styles.add(ParagraphStyle(name='Subtitle', parent=styles['Normal'],
                              fontSize=11, textColor=HexColor('#666666'),
                              spaceAfter=8*mm, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='Section', parent=styles['Heading2'],
                              fontSize=14, textColor=HexColor('#1a1a2e'),
                              spaceBefore=6*mm, spaceAfter=3*mm))
    styles.add(ParagraphStyle(name='Body2', parent=styles['Normal'],
                              fontSize=10, leading=14,
                              textColor=HexColor('#333333')))

    verdict_upper = (req.verdict or "INCONCLUSIVE").upper()
    if "FAKE" in verdict_upper:
        verdict_color = HexColor('#dc3545')
        verdict_label = "FAKE — Deepfake Detectat"
    elif "REAL" in verdict_upper:
        verdict_color = HexColor('#28a745')
        verdict_label = "REAL — Autentic"
    else:
        verdict_color = HexColor('#ffc107')
        verdict_label = "INCONCLUSIVE — Neconcludent"

    styles.add(ParagraphStyle(name='Verdict', parent=styles['Title'],
                              fontSize=20, textColor=verdict_color,
                              alignment=TA_CENTER, spaceBefore=4*mm, spaceAfter=4*mm))

    elements = []

    # Header
    elements.append(Paragraph("🛡️ DeepGuard Pro", styles['Title2']))
    elements.append(Paragraph("Raport de Analiză Forensic — Neural Multi-Agent Detection System", styles['Subtitle']))
    elements.append(HRFlowable(width="100%", color=HexColor('#cccccc'), thickness=0.5))
    elements.append(Spacer(1, 4*mm))

    # Verdict
    elements.append(Paragraph("Verdict Final", styles['Section']))
    elements.append(Paragraph(verdict_label, styles['Verdict']))
    conf_pct = min(req.confidence, 100)
    elements.append(Paragraph(f"Nivel de Încredere: <b>{conf_pct:.1f}%</b>", styles['Body2']))
    elements.append(Spacer(1, 4*mm))

    # File details table
    elements.append(Paragraph("Detalii Fișier", styles['Section']))
    meta = req.metadata or {}
    file_size_str = f"{meta.get('file_size', 0) / 1_000_000:.2f} MB" if meta.get('file_size') else "N/A"
    table_data = [
        ["Fișier:", req.filename],
        ["Dimensiune:", file_size_str],
        ["Tip MIME:", meta.get('mime_type', 'N/A')],
        ["Rezoluție:", meta.get('resolution', 'N/A')],
        ["Data Raport:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ["Generat de:", f"{current_user.get('username', 'unknown')}"],
    ]
    t = Table(table_data, colWidths=[35*mm, 130*mm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#555555')),
        ('TEXTCOLOR', (1, 0), (1, -1), HexColor('#222222')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3*mm),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 4*mm))

    # Reasoning
    elements.append(Paragraph("Analiză Neuro-Simbolică", styles['Section']))
    reasoning_text = (req.report_text or "Nu există detalii disponibile.").replace('\n', '<br/>')
    elements.append(Paragraph(reasoning_text, styles['Body2']))
    elements.append(Spacer(1, 6*mm))

    # Footer
    elements.append(HRFlowable(width="100%", color=HexColor('#cccccc'), thickness=0.5))
    elements.append(Spacer(1, 2*mm))
    footer_style = ParagraphStyle(name='Footer', parent=styles['Normal'],
                                   fontSize=8, textColor=HexColor('#999999'),
                                   alignment=TA_CENTER)
    elements.append(Paragraph(
        "Acest raport a fost generat automat de DeepGuard Pro AI. "
        "Rezultatele au caracter informativ și nu constituie o expertiză legală.",
        footer_style
    ))

    doc.build(elements)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="DeepGuard_Report_{req.filename}.pdf"'
        }
    )


# ============================================================================
# HEALTH & STATUS
# ============================================================================
@app.get("/health")
async def health():
    """Enhanced health check with per-agent model verification."""

    # Infrastructure status
    neo4j_status = "ok" if neo4j_conn.driver else "error"
    redis_status = "ok" if redis_client else "error"

    # Ping each agent's /health endpoint in parallel
    health_endpoints = {
        "vision": "http://deepguard_vision:8002/health",
        "audio":  "http://deepguard_audio:8004/health",
        "video":  "http://deepguard_video:8003/health",
        "bio":    "http://deepguard_bio:8005/health",
        "text":   "http://deepguard_text:8001/health",
    }

    async def _check_agent(name: str, url: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "status": "healthy",
                        "model_loaded": data.get("model_loaded", data.get("face_detector_loaded")),
                        "mode": data.get("mode", "unknown"),
                    }
                return {"status": "unhealthy", "error": f"HTTP {resp.status_code}"}
        except Exception as exc:
            return {"status": "unreachable", "error": str(exc)[:120]}

    agent_tasks = {name: _check_agent(name, url) for name, url in health_endpoints.items()}
    agent_results = dict(zip(agent_tasks.keys(), await asyncio.gather(*agent_tasks.values())))

    # Determine overall status
    all_agents_ok = all(a.get("status") == "healthy" for a in agent_results.values())
    overall = "healthy" if (all_agents_ok and neo4j_status == "ok" and redis_status == "ok") else "degraded"

    return {
        "status": overall,
        "neo4j": neo4j_status,
        "redis": redis_status,
        "celery": "enabled" if CELERY_ENABLED else "disabled",
        "agents": agent_results,
        "agents_count": len(AGENT_REGISTRY),
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/stats")
async def get_stats(current_user: Dict = Depends(get_current_user)):
    stats = neo4j_conn.query("""
        MATCH (j:Job)
        WITH count(j) as total,
             sum(CASE WHEN j.status = 'completed' THEN 1 ELSE 0 END) as completed,
             sum(CASE WHEN j.status = 'failed' THEN 1 ELSE 0 END) as failed,
             sum(CASE WHEN j.status IN ['queued', 'processing'] THEN 1 ELSE 0 END) as pending
        RETURN total, completed, failed, pending
    """)
    return stats[0] if stats else {"total": 0, "completed": 0, "failed": 0, "pending": 0}

@app.get("/stats/dashboard")
async def get_dashboard_stats(current_user: Dict = Depends(get_current_user)):
    """Alias pentru /stats - folosit de frontend pe pagina Dashboard."""
    username = current_user.get("username", "")
    global_stats = neo4j_conn.query("""
        MATCH (j:Job)
        WITH count(j) as total,
             sum(CASE WHEN j.status = 'completed' THEN 1 ELSE 0 END) as completed,
             sum(CASE WHEN j.status = 'failed' THEN 1 ELSE 0 END) as failed,
             sum(CASE WHEN j.status IN ['queued', 'processing'] THEN 1 ELSE 0 END) as pending
        RETURN total, completed, failed, pending
    """)
    user_stats = neo4j_conn.query("""
        MATCH (u:User {username: $u})-[:REQUESTED]->(j:Job)
        OPTIONAL MATCH (j)-[:RESULT]->(f:File)
        WITH count(j) as my_total,
             sum(CASE WHEN f.verdict = 'FAKE' THEN 1 ELSE 0 END) as my_fake,
             sum(CASE WHEN f.verdict = 'REAL' THEN 1 ELSE 0 END) as my_real
        RETURN my_total, my_fake, my_real
    """, {"u": username})
    gs = global_stats[0] if global_stats else {"total": 0, "completed": 0, "failed": 0, "pending": 0}
    us = user_stats[0] if user_stats else {"my_total": 0, "my_fake": 0, "my_real": 0}
    return {**gs, **us}

@app.post("/feedback")
async def submit_feedback(data: Dict, current_user: Dict = Depends(get_current_user)):
    """Primește feedback de la utilizator despre un rezultat de analiză."""
    username = current_user.get("username", "")
    job_id = data.get("job_id")
    rating = data.get("rating")
    comment = data.get("comment", "")
    if job_id:
        neo4j_conn.query("""
            MATCH (j:Job {id: $job_id})
            SET j.feedback_rating = $rating, j.feedback_comment = $comment, j.feedback_at = datetime()
        """, {"job_id": job_id, "rating": rating, "comment": comment})
    logger.info(f"📝 Feedback from {username}: job={job_id}, rating={rating}")
    return {"status": "ok", "message": "Feedback înregistrat"}

# ============================================================================
# STATS REPORT (pentru n8n Cron job zilnic)
# ============================================================================
from fastapi import Query as QueryParam

@app.get("/stats/report")
async def get_daily_report(api_key: str = QueryParam(...)):
    """Endpoint public (protejat cu API key) pentru raport zilnic n8n."""
    report_api_key = os.getenv("REPORT_API_KEY", "deepguard_report_2025")
    if api_key != report_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    stats = neo4j_conn.query("""
        MATCH (j:Job)
        WHERE j.status = 'completed'
        OPTIONAL MATCH (j)-[:RESULT]->(f:File)
        OPTIONAL MATCH (u:User)-[:REQUESTED]->(j)
        WITH j, f, u
        WHERE j.completed_at IS NOT NULL
        RETURN
            count(j) as total,
            sum(CASE WHEN f.verdict = 'FAKE' THEN 1 ELSE 0 END) as fake_count,
            sum(CASE WHEN f.verdict = 'REAL' THEN 1 ELSE 0 END) as real_count,
            sum(CASE WHEN f.verdict = 'INCONCLUSIVE' THEN 1 ELSE 0 END) as inconclusive_count,
            avg(f.confidence) as avg_confidence,
            collect(DISTINCT u.username) as users
    """)

    if not stats:
        return {"period": "all_time", "total": 0, "fake_count": 0, "real_count": 0,
                "inconclusive_count": 0, "avg_confidence": 0, "users": []}

    data = stats[0]
    total = data.get("total") or 0
    fake_count = data.get("fake_count") or 0
    real_count = data.get("real_count") or 0
    inconclusive_count = data.get("inconclusive_count") or 0
    avg_conf = data.get("avg_confidence") or 0
    users = [u for u in (data.get("users") or []) if u]

    return {
        "period": "all_time",
        "total_analyses": total,
        "fake_detected": fake_count,
        "real_confirmed": real_count,
        "inconclusive": inconclusive_count,
        "fake_percentage": round((fake_count * 100.0 / total) if total > 0 else 0, 1),
        "real_percentage": round((real_count * 100.0 / total) if total > 0 else 0, 1),
        "average_confidence": round(float(avg_conf) * 100 if avg_conf else 0, 1),
        "active_users": users,
        "total_users": len(users),
        "generated_at": datetime.now().isoformat()
    }

# ============================================================================
# OSINT REPORT CALLBACK  (n8n → DeepGuard)
# ============================================================================
class OSINTReportPayload(BaseModel):
    job_id: str
    osint_results: Any
    investigation_completed: str

@app.post("/osint/report")
async def receive_osint_report(payload: OSINTReportPayload, request: Request):
    """
    Callback endpoint apelat de n8n după finalizarea investigației OSINT.
    Salvează rezultatele OSINT în Neo4j și actualizează job-ul.
    Protejat cu X-Api-Key header.
    """
    # Validare API key
    report_api_key = os.getenv("REPORT_API_KEY", "deepguard_report_2025")
    api_key_header = request.headers.get("X-Api-Key", "")
    if api_key_header != report_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    job_id = payload.job_id
    osint_results = payload.osint_results
    completed_at = payload.investigation_completed

    # Parsare osint_results dacă e string JSON
    if isinstance(osint_results, str):
        try:
            osint_results = json.loads(osint_results)
        except Exception:
            pass

    try:
        neo4j_conn.query("""
            MATCH (j:Job {id: $job_id})
            SET j.osint_results = $osint_results,
                j.osint_completed_at = $completed_at,
                j.osint_risk_level = $risk_level
        """, {
            "job_id": job_id,
            "osint_results": json.dumps(osint_results),
            "completed_at": completed_at,
            "risk_level": osint_results.get("risk_level", "UNKNOWN") if isinstance(osint_results, dict) else "UNKNOWN"
        })

        logger.info(f"✅ OSINT results saved for job {job_id} — risk: {osint_results.get('risk_level') if isinstance(osint_results, dict) else 'N/A'}")

        return {
            "status": "ok",
            "job_id": job_id,
            "osint_risk_level": osint_results.get("risk_level") if isinstance(osint_results, dict) else "UNKNOWN"
        }

    except Exception as e:
        logger.error(f"❌ Failed to save OSINT results for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save OSINT results: {str(e)}")

# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================
@app.get("/admin/users")
async def admin_get_users(current_user: Dict = Depends(require_admin)):
    """Lista tuturor utilizatorilor (doar admin)."""
    results = neo4j_conn.query("""
        MATCH (u:User)
        OPTIONAL MATCH (u)-[:REQUESTED]->(j:Job)
        RETURN u.username as username, u.email as email, u.role as role,
               u.created_at as created_at, count(j) as total_jobs
        ORDER BY u.created_at DESC
    """)
    return {"users": results or []}

@app.get("/admin/jobs")
async def admin_get_jobs(current_user: Dict = Depends(require_admin), limit: int = 50, offset: int = 0):
    """Toate job-urile din sistem (doar admin)."""
    results = neo4j_conn.query("""
        MATCH (u:User)-[:REQUESTED]->(j:Job)
        OPTIONAL MATCH (j)-[:RESULT]->(f:File)
        RETURN j.id as job_id, j.status as status, j.progress as progress,
               j.filename as filename, j.created_at as created_at,
               j.completed_at as completed_at, f.verdict as verdict,
               f.confidence as confidence, u.username as username
        ORDER BY j.created_at DESC
        SKIP $offset LIMIT $limit
    """, {"offset": offset, "limit": limit})
    return {"jobs": results or [], "limit": limit, "offset": offset}

@app.get("/admin/stats")
async def admin_get_stats(current_user: Dict = Depends(require_admin)):
    """Statistici globale pentru admin dashboard."""
    stats = neo4j_conn.query("""
        MATCH (j:Job)
        OPTIONAL MATCH (j)-[:RESULT]->(f:File)
        RETURN
            count(j) as total_jobs,
            sum(CASE WHEN j.status = 'completed' THEN 1 ELSE 0 END) as completed,
            sum(CASE WHEN j.status = 'failed' THEN 1 ELSE 0 END) as failed,
            sum(CASE WHEN j.status IN ['queued', 'processing'] THEN 1 ELSE 0 END) as pending,
            sum(CASE WHEN f.verdict = 'FAKE' THEN 1 ELSE 0 END) as fake_count,
            sum(CASE WHEN f.verdict = 'REAL' THEN 1 ELSE 0 END) as real_count,
            sum(CASE WHEN f.verdict = 'INCONCLUSIVE' THEN 1 ELSE 0 END) as inconclusive_count,
            avg(f.confidence) as avg_confidence
    """)
    users_count = neo4j_conn.query("MATCH (u:User) RETURN count(u) as total")
    total_users = users_count[0].get("total", 0) if users_count else 0

    data = stats[0] if stats else {}
    return {
        "total_jobs": data.get("total_jobs") or 0,
        "completed": data.get("completed") or 0,
        "failed": data.get("failed") or 0,
        "pending": data.get("pending") or 0,
        "fake_count": data.get("fake_count") or 0,
        "real_count": data.get("real_count") or 0,
        "inconclusive_count": data.get("inconclusive_count") or 0,
        "avg_confidence": round(float(data.get("avg_confidence") or 0) * 100, 1),
        "total_users": total_users,
        "generated_at": datetime.now().isoformat()
    }

@app.delete("/admin/users/{username}")
async def admin_delete_user(username: str, current_user: Dict = Depends(require_admin)):
    """Șterge un utilizator (doar admin, nu se poate șterge pe sine)."""
    if username == current_user.get("username"):
        raise HTTPException(status_code=400, detail="Nu te poți șterge pe tine însuți")
    if username == "admin":
        raise HTTPException(status_code=400, detail="Nu se poate șterge userul admin")
    existing = neo4j_conn.query("MATCH (u:User {username: $u}) RETURN u", {"u": username})
    if not existing:
        raise HTTPException(status_code=404, detail="User not found")
    neo4j_conn.query("""
        MATCH (u:User {username: $u})
        OPTIONAL MATCH (u)-[:REQUESTED]->(j:Job)
        DETACH DELETE u, j
    """, {"u": username})
    return {"message": f"User '{username}' deleted successfully"}

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")