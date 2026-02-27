# pyright: reportMissingImports=false
"""
DeepGuard Pro - Celery Background Tasks
"""

import os
import sys
import json
import logging
import asyncio
import httpx
import base64
import tempfile
from typing import Dict, List, Optional, Any, cast
from datetime import datetime
from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
import mimetypes
import subprocess
from pathlib import Path
try:
    import cv2
    from PIL import Image
    METADATA_EXTRACTION_AVAILABLE = True
except ImportError:
    METADATA_EXTRACTION_AVAILABLE = False
    logging.warning("⚠️ OpenCV or PIL not available - metadata extraction disabled")

# ============================================================================
# SETUP PATHS
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# ============================================================================
# CELERY APP CONFIGURATION
# ============================================================================
_redis_pass = os.getenv('REDIS_PASSWORD', 'deepguard_redis_secret')
_default_redis_url = f'redis://:{_redis_pass}@redis:6379/0'
celery_app = Celery(
    'deepguard_tasks',
    broker=os.getenv('CELERY_BROKER_URL', _default_redis_url),
    backend=os.getenv('CELERY_RESULT_BACKEND', _default_redis_url)
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3300,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    broker_connection_retry_on_startup=True,
    result_expires=3600,
)

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepGuard.Celery")

# ============================================================================
# IMPORTS (After Celery setup to avoid circular dependencies)
# ============================================================================
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.error("❌ Neo4j driver missing")

try:
    from email_notifier import EmailNotifier  # type: ignore
    email_tool = EmailNotifier()
except ImportError:
    email_tool = None
    logger.warning("⚠️ EmailNotifier not available")

try:
    from neurosymbolic import NeuroSymbolicDebater  # type: ignore
    DEBATER_AVAILABLE = True
except ImportError:
    DEBATER_AVAILABLE = False
    logger.warning("⚠️ NeuroSymbolicDebater not available")

try:
    import s3_manager as s3_lib  # type: ignore
    os.environ["S3_ENDPOINT"] = os.getenv("S3_ENDPOINT", "http://seaweedfs:8333")
    s3_manager = s3_lib.S3Manager()
except Exception as e:
    s3_manager = None
    logger.warning(f"⚠️ S3Manager not available: {e}")

try:
    from tools.synthid_sentinel import SynthIDSentinel  # type: ignore
    synthid_tool = SynthIDSentinel()
except ImportError:
    synthid_tool = None
    logger.warning("⚠️ SynthID Sentinel not available")

try:
    from gpt4_vision_analyzer import active_vision_analyzer as gemini_vision
    GEMINI_VISION_AVAILABLE = True
    logger.info(f"✅ Vision Analyzer loaded: {type(gemini_vision).__name__}")
except ImportError:
    try:
        from gemini_vision_analyzer import gemini_vision
        GEMINI_VISION_AVAILABLE = True
        logger.info("✅ Vision Analyzer loaded: GeminiVisionAnalyzer (fallback)")
    except ImportError:
        GEMINI_VISION_AVAILABLE = False
        logger.warning("⚠️ Vision Analyzer (GPT-4o / Gemini) not available")

# ============================================================================
# NEO4J CONNECTION
# ============================================================================
class Neo4jConnection:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "deepguard_secret")
        self.driver = None
        if NEO4J_AVAILABLE:
            self._connect()

    def _connect(self):
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=120
            )
            self.driver.verify_connectivity()
            logger.info("✅ Neo4j connected (Celery Worker)")
        except Exception as e:
            logger.error(f"❌ Neo4j Connection Error: {e}")
            self.driver = None

    def query(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.driver:
            logger.warning("⚠️ Neo4j driver not active, attempting reconnection...")
            self._connect()
        if not self.driver:
            logger.error("❌ Neo4j driver not initialized after retry")
            return []
        try:
            with self.driver.session() as session:
                # FIX pentru Pylance: cast la Any pentru a evita eroarea de tip
                result = session.run(cast(Any, cypher), parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"❌ Neo4j Query failed: {e}")
            if "ServiceUnavailable" in str(e) or "SessionExpired" in str(e):
                self.driver = None
            return []

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("✅ Neo4j connection closed")

# Global Neo4j connection
neo4j_conn = Neo4jConnection()

# ============================================================================
# REDIS PUBLISH (for WebSocket status updates)
# ============================================================================
def publish_status_to_redis(message: str, status: str = "info", job_id: Optional[str] = None, progress: int = 0):
    """Publish status to Redis so WebSocket clients receive real-time updates"""
    try:
        import redis
        r = redis.from_url(os.getenv("CELERY_BROKER_URL", _default_redis_url))
        payload = json.dumps({
            "type": "status",
            "status": status,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "progress": progress
        })
        r.publish("deepguard_status_updates", payload)
    except Exception as e:
        logger.warning(f"Redis publish failed: {e}")

# ============================================================================
# AGENT REGISTRY
# ============================================================================
AGENT_REGISTRY = {
    "vision": "http://deepguard_vision:8002/analyze",
    "audio": "http://deepguard_audio:8004/analyze",
    "video": "http://deepguard_video:8003/analyze",
    "bio": "http://deepguard_bio:8005/analyze",
    "text": "http://deepguard_text:8001/analyze"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def update_job_status(job_id: str, status: str, progress: int = 0, message: str = "", extra_data: Optional[Dict] = None):
    """Update job status in Neo4j with progress tracking + Redis broadcast for WebSocket"""
    params = {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if extra_data:
        params.update(extra_data)
    
    neo4j_conn.query("""
        MATCH (j:Job {id: $job_id})
        SET j.status = $status, 
            j.progress = $progress, 
            j.last_update = datetime($timestamp), 
            j.status_message = $message
    """, params)
    
    # Broadcast to WebSocket clients via Redis
    publish_status_to_redis(message, "thinking" if status == "processing" else ("success" if status == "completed" else "info"), job_id, progress)
    
    logger.info(f"📊 Job {job_id}: {status} ({progress}%) - {message}")


async def call_agent_async(name: str, endpoint: str, content: bytes, filename: str) -> Dict[str, Any]:
    """Async HTTP call to agent service.
    Text agent expects JSON {text: ...}, all others expect multipart file upload.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            logger.info(f"🔄 Calling {name} agent at {endpoint}")

            if name == "text":
                # Text agent expects JSON body with 'text' field
                try:
                    text_content = content.decode("utf-8", errors="replace")
                except Exception:
                    text_content = content.decode("latin-1", errors="replace")
                response = await client.post(endpoint, json={"text": text_content})
            else:
                files = {'file': (filename, content, 'application/octet-stream')}
                response = await client.post(endpoint, files=files)

            response.raise_for_status()

            result = response.json()
            logger.info(f"✅ {name} agent response: {result.get('verdict', 'N/A')}")
            return result

    except httpx.TimeoutException:
        logger.error(f"⏱️ Timeout calling {name} agent")
        return {"agent": name, "verdict": "ERROR", "error": "Timeout after 120s"}
    except httpx.HTTPError as e:
        logger.error(f"❌ HTTP error calling {name}: {e}")
        return {"agent": name, "verdict": "ERROR", "error": str(e)}
    except Exception as e:
        logger.error(f"❌ Unexpected error calling {name}: {e}")
        return {"agent": name, "verdict": "ERROR", "error": str(e)}


VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.wmv']


def extract_audio_from_video(video_content: bytes, file_type: str) -> Optional[bytes]:
    """Extract audio track from video using ffmpeg. Returns WAV bytes or None."""
    tmp_video_path = None
    tmp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp_video:
            tmp_video.write(video_content)
            tmp_video_path = tmp_video.name

        tmp_audio_path = tmp_video_path + ".wav"

        result = subprocess.run(
            ["ffmpeg", "-i", tmp_video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", tmp_audio_path, "-y"],
            capture_output=True, timeout=60
        )

        if result.returncode == 0 and os.path.exists(tmp_audio_path):
            with open(tmp_audio_path, "rb") as f:
                audio_bytes = f.read()
            if len(audio_bytes) > 1000:  # Sanity check: at least ~1KB of audio
                logger.info(f"🎵 Audio extracted: {len(audio_bytes)} bytes")
                return audio_bytes
            else:
                logger.warning("⚠️ Extracted audio too small — video may have no audio track")
                return None

        stderr = result.stderr.decode(errors='replace') if result.stderr else ""
        logger.warning(f"⚠️ ffmpeg audio extraction failed (rc={result.returncode}): {stderr[:200]}")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("⚠️ ffmpeg audio extraction timed out")
        return None
    except Exception as e:
        logger.warning(f"⚠️ Audio extraction failed: {e}")
        return None
    finally:
        if tmp_video_path and os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)


def select_agents_by_filetype(file_type: str, filename: str) -> List[str]:
    """
    Smart agent selection based on file extension.
    CRITICAL: Vision agent uses PIL and can ONLY process images - NEVER send videos to it.
    """
    ext = file_type.lower()
    name_lower = (filename or "").lower()
    
    # Image files → ONLY vision agent (PIL-based, cannot handle video)
    if any(x in ext for x in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff']):
        logger.info(f"📸 Image detected ({ext}). Routing to: vision agent ONLY")
        return ["vision"]
    
    # Video files → video + bio + audio (audio track extracted by pipeline via ffmpeg)
    elif any(x in ext for x in VIDEO_EXTENSIONS):
        logger.info(f"🎬 Video detected ({ext}). Routing to: video, bio, audio agents")
        return ["video", "bio", "audio"]
    
    # Audio files
    elif any(x in ext for x in ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']):
        logger.info(f"🎙️ Audio detected ({ext}). Routing to: audio agent")
        return ["audio"]
    
    # Text files
    elif any(x in ext for x in ['.txt', '.pdf', '.doc', '.docx', '.md']):
        logger.info(f"📄 Text detected ({ext}). Routing to: text agent")
        return ["text"]
    
    # Default fallback (unknown type - avoid vision to prevent PIL crash)
    else:
        logger.warning(f"⚠️ Unknown file type ({ext}). Routing to: video, bio (no vision)")
        return ["video", "bio"]


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================
async def run_analysis_pipeline(
    job_id: str, 
    username: str, 
    f_hash: str, 
    filename: str, 
    content: bytes, 
    file_type: str, 
    query: Optional[str]
) -> Dict[str, Any]:
    """
    Main analysis pipeline - orchestrates all agents and logic
    """
    
    # ========================================================================
    # STEP 1: Agent Selection (30%)
    # ========================================================================
    update_job_status(job_id, "processing", 30, "Selectare agenți AI...")
    
    routing_reason = ""
    try:
        from agentic_router import select_agents as agentic_select_agents
        selected_agents, routing_reason = agentic_select_agents(file_type, filename, query)
        logger.info(f"🎯 Agentic routing: {selected_agents} — {routing_reason}")
    except Exception as e:
        logger.warning(f"⚠️ Agentic router failed ({e}), using file-type fallback")
        selected_agents = select_agents_by_filetype(file_type, filename)
        routing_reason = "Fallback: file-type routing"
    logger.info(f"🎯 Selected agents for {filename}: {selected_agents}")
    
    # ========================================================================
    # STEP 2: SynthID Watermark Check (40%)
    # ========================================================================
    synthid_result = None
    if synthid_tool and hasattr(synthid_tool, 'enabled') and synthid_tool.enabled:
        if any(x in file_type.lower() for x in ['image', 'video', 'jpg', 'png', 'mp4', 'jpeg']):
            try:
                update_job_status(job_id, "processing", 40, "Verificare watermark digital (SynthID)...")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                synthid_result = await synthid_tool.verify_media(tmp_path)
                os.remove(tmp_path)
                
                logger.info(f"🔍 SynthID result: {synthid_result}")
                
            except Exception as e:
                logger.error(f"❌ SynthID verification failed: {e}")
    
    # ========================================================================
    # STEP 3: Call Agent Services (50%)
    # ========================================================================
    update_job_status(job_id, "processing", 50, f"Analiză cu {len(selected_agents)} agenți neurali...")

    # For video files: extract audio track for audio agent
    is_video = any(x in file_type.lower() for x in VIDEO_EXTENSIONS)
    audio_content = None
    if is_video and "audio" in selected_agents:
        update_job_status(job_id, "processing", 45, "🎵 Extragere audio din video...")
        audio_content = extract_audio_from_video(content, file_type)
        if audio_content is None:
            logger.warning("⚠️ No audio extracted — removing audio agent from pipeline")
            selected_agents = [a for a in selected_agents if a != "audio"]

    # Build parallel tasks — audio agent gets extracted audio, others get original content
    tasks = []
    actual_agents = []
    for agent in selected_agents:
        if agent not in AGENT_REGISTRY:
            continue
        if agent == "audio" and is_video and audio_content is not None:
            audio_filename = filename.rsplit('.', 1)[0] + ".wav"
            tasks.append(call_agent_async(agent, AGENT_REGISTRY[agent], audio_content, audio_filename))
        else:
            tasks.append(call_agent_async(agent, AGENT_REGISTRY[agent], content, filename))
        actual_agents.append(agent)

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results (handle exceptions)
    agent_results = {}
    for i, result in enumerate(raw_results):
        agent_name = actual_agents[i]
        if isinstance(result, Exception):
            logger.error(f"❌ Agent {agent_name} failed: {result}")
            agent_results[agent_name] = {
                "agent": agent_name,
                "verdict": "ERROR",
                "error": str(result)
            }
        else:
            agent_results[agent_name] = result
    
    logger.info(f"📊 Agent results: {json.dumps(agent_results, indent=2)}")
    
    # ========================================================================
    # STEP 3.5: Vision Analysis — GPT-4o (primary) / Gemini (fallback) (65%)
    # ========================================================================
    vision_analyzer_name = type(gemini_vision).__name__ if GEMINI_VISION_AVAILABLE else "N/A"
    gemini_analysis = None
    if GEMINI_VISION_AVAILABLE and gemini_vision.enabled:
        update_job_status(job_id, "processing", 65, f"🤖 {vision_analyzer_name} (frame-by-frame analysis)...")

        try:
            # Salvează conținutul într-un fișier temporar
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # Analizează în funcție de tip
            if any(x in file_type.lower() for x in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']):
                # Video: analiză multi-frame with progress 65% -> 90%
                def on_gemini_frame_progress(jid: str, current: int, total: int) -> None:
                    progress = 65 + int(25 * current / total) if total else 65
                    update_job_status(jid, "processing", progress, f"🤖 Vision frame {current}/{total}...")
                logger.info(f"🎬 Starting {vision_analyzer_name} multi-frame video analysis...")
                gemini_analysis = gemini_vision.analyze_video_multiframe(
                    tmp_path, job_id, on_frame_progress=on_gemini_frame_progress
                )

            elif any(x in file_type.lower() for x in ['.jpg', '.png', '.jpeg', '.webp', '.gif', '.bmp']):
                # Imagine: analiză single frame
                logger.info(f"🖼️ Starting {vision_analyzer_name} image analysis...")
                gemini_analysis = gemini_vision.analyze_image(tmp_path, job_id)

            os.remove(tmp_path)  # Cleanup

            if gemini_analysis and 'error' not in gemini_analysis:
                agent_results["gemini_vision"] = gemini_analysis
                logger.info(f"🤖 {vision_analyzer_name}: {gemini_analysis.get('verdict', 'N/A')} ({gemini_analysis.get('confidence', 0):.0%})")

                # Dacă vision analyzer detectează FAKE cu confidență mare, prioritizăm verdictul lui
                if gemini_analysis.get('verdict') == 'FAKE' and gemini_analysis.get('confidence', 0) > 0.8:
                    logger.info(f"⚠️ {vision_analyzer_name} detected high-confidence FAKE - will prioritize this verdict")
            else:
                logger.warning(f"⚠️ {vision_analyzer_name} analysis failed or returned error")

        except Exception as e:
            logger.error(f"❌ Vision analysis error ({vision_analyzer_name}): {e}", exc_info=True)
    
    # ========================================================================
    # STEP 4: Neuro-Symbolic Reasoning (80%)
    # ========================================================================
    update_job_status(job_id, "processing", 80, "Raționament logic neuro-simbolic...")
    
    final_verdict = "INCONCLUSIVE"
    reasoning = "Sistemul logic nu este disponibil."
    confidence = 0.0
    
    # Priority 1: SynthID Override (Ground Truth)
    if synthid_result and synthid_result.get("detected"):
        final_verdict = "FAKE"
        reasoning = "🔴 **Google SynthID Watermark Detectat**: Conținutul este CERTIFICAT ca fiind generat artificial (AI-Generated). Aceasta este dovada digitală invizibilă integrată în timpul generării."
        confidence = 0.99
        agent_results["synthid"] = synthid_result
        logger.info("🚨 SynthID OVERRIDE: Content is AI-Generated")
    
    # Priority 2: Gemini Vision High Confidence
    elif gemini_analysis and gemini_analysis.get('verdict') == 'FAKE' and gemini_analysis.get('confidence', 0) > 0.8:
        final_verdict = "FAKE"
        reasoning = gemini_analysis.get('reasoning', 'Gemini Vision a detectat artefacte AI-generated.')
        confidence = gemini_analysis.get('confidence', 0.85)
        logger.info("🤖 Gemini Vision OVERRIDE: High-confidence FAKE detection")
    
    # Priority 3: Neuro-Symbolic Debater
    elif DEBATER_AVAILABLE:
        try:
            from neurosymbolic import NeuroSymbolicDebater  # type: ignore
            
            if synthid_result:
                agent_results["synthid"] = synthid_result
            
            debater = NeuroSymbolicDebater()
            logic_result = debater.validate_contradictions(agent_results)
            
            final_verdict = logic_result.get("final_verdict", "INCONCLUSIVE")
            reasoning = logic_result.get("reasoning", "Nu s-au detectat inconsistențe logice.")
            confidence = logic_result.get("confidence", 0.0)
            
            logger.info(f"🧠 Debater verdict: {final_verdict} ({confidence:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Debater failed: {e}")
            reasoning = f"Eroare în modulul logic: {str(e)}"
    
    # Priority 4: Gemini Vision as fallback (skip when confidence ~50% to avoid unhelpful "REAL (50%)")
    elif gemini_analysis and gemini_analysis.get('verdict') in ['FAKE', 'SUSPICIOUS', 'REAL']:
        gemini_conf = gemini_analysis.get('confidence', 0.0)
        if 0.45 <= gemini_conf <= 0.55:
            logger.info("🤖 Skipping Gemini Vision as decisive (confidence ~50%, treated as INCONCLUSIVE)")
        else:
            final_verdict = gemini_analysis.get('verdict', 'INCONCLUSIVE')
            reasoning = gemini_analysis.get('reasoning', 'Analiză Gemini Vision.')
            confidence = gemini_conf
            logger.info(f"🤖 Using Gemini Vision verdict: {final_verdict}")
    
    # ========================================================================
    # STEP 5: Upload to S3 Storage (90%)
    # ========================================================================
    update_job_status(job_id, "processing", 90, "Salvare în cloud storage...")
    
    s3_url = ""
    if s3_manager and content:
        try:
            s3_url = s3_manager.upload_file(content, f"{f_hash}_{filename}")
            logger.info(f"☁️ Uploaded to S3: {s3_url}")
        except Exception as e:
            logger.error(f"❌ S3 upload failed: {e}")
    
    # ========================================================================
    # STEP 6: Save Results to Neo4j (95%)
    # ========================================================================
    update_job_status(job_id, "processing", 95, "Salvare rezultate...")
    
    full_result = {
        "verdict": final_verdict,
        "reasoning": reasoning,
        "confidence": confidence,
        "agents_used": selected_agents,
        "details": agent_results,
        "timestamp": datetime.utcnow().isoformat(),
        "file_hash": f_hash,
        "filename": filename
    }
    
    # Adaugă link-uri către frame-uri annotate dacă există
    if gemini_analysis and gemini_analysis.get('annotated_frames'):
        full_result['annotated_frames'] = gemini_analysis['annotated_frames']
    
    serialized_result = json.dumps(full_result)
    
    neo4j_params = {
        "job_id": job_id,
        "u": username,
        "hash": f_hash,
        "verdict": final_verdict,
        "reason": reasoning,
        "confidence": confidence,
        "url": s3_url,
        "filename": filename,
        "serialized_result": serialized_result
    }
    
    # Primary: Full query with User relationship
    primary_saved = False
    try:
        neo4j_conn.query("""
            MATCH (u:User {username: $u})-[:REQUESTED]->(j:Job {id: $job_id})
            MERGE (f:File {hash: $hash})
            SET f.verdict = $verdict,
                f.reasoning = $reason,
                f.confidence = $confidence,
                f.url = $url,
                f.timestamp = datetime(),
                f.filename = $filename
            SET j.status = 'completed',
                j.result = $serialized_result,
                j.progress = 100,
                j.completed_at = datetime()
            MERGE (u)-[:OWNS]->(f)
            MERGE (j)-[:RESULT]->(f)
        """, neo4j_params)
        primary_saved = True
        logger.info("✅ Neo4j primary save succeeded")
    except Exception as e:
        logger.warning(f"Primary Neo4j save failed: {e}")

    # Fallback: ONLY if primary failed - Ensure File + RESULT exist
    if not primary_saved:
        try:
            neo4j_conn.query("""
                MATCH (j:Job {id: $job_id})
                MERGE (f:File {hash: $hash})
                SET f.verdict = $verdict,
                    f.reasoning = $reason,
                    f.confidence = $confidence,
                    f.url = $url,
                    f.timestamp = datetime(),
                    f.filename = $filename
                SET j.status = 'completed',
                    j.result = $serialized_result,
                    j.progress = 100,
                    j.completed_at = datetime()
                MERGE (j)-[:RESULT]->(f)
            """, neo4j_params)
            logger.info("✅ Neo4j fallback save succeeded")
        except Exception as e:
            logger.error(f"❌ Neo4j fallback save failed: {e}")
    
    # ========================================================================
    # STEP 7: Send Email Notification (100%)
    # ========================================================================
    update_job_status(job_id, "completed", 100, "✅ Analiză finalizată!")
    
    # STEP 7: Ensure we have email_tool (lazy init if import failed at worker startup)
    _email_tool = email_tool
    if _email_tool is None:
        try:
            from email_notifier import EmailNotifier  # type: ignore
            _email_tool = EmailNotifier()
        except Exception as e:
            logger.debug(f"EmailNotifier lazy init failed: {e}")
    if _email_tool:
        try:
            if not username or username.strip().lower() in ("anonymous", ""):
                logger.info(f"📧 Skipping completion email: no authenticated user (job: {job_id})")
            else:
                user_data = neo4j_conn.query(
                    "MATCH (u:User {username: $u}) RETURN u.email as email",
                    {"u": username.strip()}
                )
                if not user_data:
                    logger.warning(f"📧 No User found for username={username!r} — skipping completion email (job: {job_id})")
                elif not user_data[0].get('email'):
                    logger.warning(f"📧 User {username!r} has no email in DB — skipping completion email (job: {job_id})")
                else:
                    user_email = str(user_data[0]['email']).strip()
                    if not user_email:
                        logger.warning(f"📧 User {username!r} email is empty — skipping (job: {job_id})")
                    else:
                        logger.info(f"📧 Sending completion email to {user_email} (job: {job_id})")
                        _email_tool.send_analysis_complete(
                            user_email,
                            filename,
                            final_verdict,
                            reasoning,
                            job_id=job_id,
                            confidence=confidence * 100 if confidence <= 1.0 else confidence
                        )
        except Exception as e:
            logger.error(f"❌ Email notification failed: {e}", exc_info=True)
    else:
        logger.warning("📧 Email not sent: EmailNotifier not available (check MAIL_* env and email_notifier import)")
    
    logger.info(f"✅ Job {job_id} COMPLETED: {final_verdict} ({confidence:.2f})")
    
    return {
        "job_id": job_id,
        "verdict": final_verdict,
        "reasoning": reasoning,
        "confidence": confidence,
        "status": "completed"
    }


# ============================================================================
# CELERY TASK DEFINITION
# ============================================================================
@celery_app.task(
    bind=True,
    name='process_media_analysis',
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True
)
def process_media_analysis(
    self,
    job_id: str,
    username: str,
    f_hash: str,
    filename: str,
    content_b64: str,
    file_type: str,
    query: Optional[str] = None
):
    """
    🚀 Celery Task: Process Media Analysis
    
    This task runs in the background and handles the entire analysis pipeline.
    The content is base64-encoded to be serializable in Redis.
    """
    
    logger.info(f"🎬 Starting task for job {job_id} (attempt {self.request.retries + 1})")
    
    # Decode base64 content
    try:
        content = base64.b64decode(content_b64)
    except Exception as e:
        logger.error(f"❌ Failed to decode content: {e}")
        update_job_status(job_id, "failed", 0, f"Eroare decodare fișier: {str(e)}")
        raise
    
    # Update status to processing
    update_job_status(job_id, "processing", 10, "Extracting metadata & Initializing...")

    # METADATA EXTRACTION
    resolution = "Unknown"
    file_size = len(content)
    mime_type = "application/octet-stream"
    
    try:
        # Detect MIME type
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        mime_type, _ = mimetypes.guess_type(tmp_path)
        if not mime_type:
            mime_type = "application/octet-stream"

        if METADATA_EXTRACTION_AVAILABLE:
            try:
                if mime_type.startswith("video") or file_type.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    cap = cv2.VideoCapture(tmp_path)
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        resolution = f"{width}x{height}"
                        cap.release()
                elif mime_type.startswith("image") or file_type.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    img = Image.open(tmp_path)
                    resolution = f"{img.width}x{img.height}"
                    img.close()
            except Exception as e:
                logger.warning(f"⚠️ Metadata extraction failed: {e}")
        
        # Cleanup metadata temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        # Update Job node with metadata
        neo4j_conn.query("""
            MATCH (j:Job {id: $job_id})
            SET j.file_size = $file_size,
                j.mime_type = $mime_type,
                j.resolution = $resolution
        """, {
            "job_id": job_id,
            "file_size": file_size,
            "mime_type": mime_type,
            "resolution": resolution
        })
        logger.info(f"📋 Metadata saved: {resolution}, {file_size} bytes, {mime_type}")

    except Exception as e:
        logger.error(f"❌ Metadata step failed: {e}")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Run async pipeline in sync context
        result = loop.run_until_complete(
            run_analysis_pipeline(
                job_id, username, f_hash, filename,
                content, file_type, query
            )
        )

        logger.info(f"✅ Task completed successfully: {result}")
        return result

    except Exception as e:
        logger.error(f"❌ Task failed for job {job_id}: {e}", exc_info=True)

        update_job_status(job_id, "failed", 0, f"Eroare: {str(e)}")

        neo4j_conn.query("""
            MATCH (j:Job {id: $job_id})
            SET j.status = 'failed', j.error = $error, j.failed_at = datetime()
        """, {"job_id": job_id, "error": str(e)})

        raise
    finally:
        loop.close()


# ============================================================================
# CELERY SIGNALS (Monitoring & Cleanup)
# ============================================================================
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """Called before task execution"""
    if task:
        logger.info(f"⏳ Task {task.name} [{task_id}] starting...")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, **extra):
    """Called after successful task execution"""
    if task:
        logger.info(f"✅ Task {task.name} [{task_id}] completed successfully")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, args=None, kwargs=None, traceback=None, einfo=None, **extra):
    """Called when task fails — marks job as FAILED in Neo4j so it doesn't stay stuck in 'processing'"""
    if sender:
        logger.error(f"❌ Task {sender.name} [{task_id}] FAILED: {exception}")

        # Extract job_id from task args to update Neo4j
        try:
            job_id = None
            if args and len(args) >= 1:
                job_id = args[0]  # First positional arg is job_id
            elif kwargs:
                job_id = kwargs.get('job_id')

            if job_id:
                error_msg = str(exception)[:500] if exception else "Unknown error"
                neo4j_conn.query("""
                    MATCH (j:Job {id: $job_id})
                    SET j.status = 'failed',
                        j.error = $error,
                        j.failed_at = datetime(),
                        j.progress = 0,
                        j.status_message = $msg
                """, {
                    "job_id": job_id,
                    "error": error_msg,
                    "msg": f"Analiză eșuată: {error_msg[:100]}"
                })
                publish_status_to_redis(
                    f"❌ Analiza a eșuat: {error_msg[:100]}",
                    "error", job_id, 0
                )
                logger.info(f"📝 Job {job_id} marked as FAILED in Neo4j")
        except Exception as e:
            logger.error(f"❌ Failed to update job status on failure: {e}")


# ============================================================================
# PERIODIC TASKS (Cleanup, Maintenance)
# ============================================================================
@celery_app.task(name='cleanup_old_jobs')
def cleanup_old_jobs():
    """Clean up completed jobs older than 7 days"""
    try:
        result = neo4j_conn.query("""
            MATCH (j:Job)
            WHERE j.completed_at < datetime() - duration('P7D')
            AND j.status = 'completed'
            DETACH DELETE j
            RETURN count(j) as deleted
        """)
        
        count = result[0].get('deleted', 0) if result else 0
        logger.info(f"🧹 Cleaned up {count} old jobs")
        
        return {"deleted": count}
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return {"error": str(e)}


# ============================================================================
# CELERY BEAT SCHEDULE (Periodic Tasks)
# ============================================================================
celery_app.conf.beat_schedule = {
    'cleanup-old-jobs-daily': {
        'task': 'cleanup_old_jobs',
        'schedule': 86400.0,  # Every 24 hours
    },
}

# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================
def cleanup_on_shutdown():
    """Cleanup resources on worker shutdown"""
    logger.info("🛑 Shutting down Celery worker...")
    neo4j_conn.close()

import atexit
atexit.register(cleanup_on_shutdown)