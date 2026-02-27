"""
Vertex AI Helper - Optional Google Cloud Integration
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger("DeepGuard.VertexAI")

class VertexAIManager:
    def __init__(self):
        self.enabled = os.getenv("ENABLE_VERTEX_AI", "false").lower() == "true"
        
        if not self.enabled:
            logger.info("ℹ️ Vertex AI disabled")
            return
        
        try:
            from google.cloud import aiplatform
            from google.oauth2 import service_account
            
            project_id = os.getenv("VERTEX_AI_PROJECT_ID")
            location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            
            if not project_id:
                logger.warning("⚠️ VERTEX_AI_PROJECT_ID not set")
                self.enabled = False
                return
            
            credentials = None
            if creds_path and os.path.exists(creds_path):
                credentials = service_account.Credentials.from_service_account_file(creds_path)
            
            aiplatform.init(project=project_id, location=location, credentials=credentials)
            logger.info(f"✅ Vertex AI initialized: {project_id}")
            
        except Exception as e:
            logger.error(f"❌ Vertex AI init failed: {e}")
            self.enabled = False
    
    def select_agents(self, file_type: str) -> List[str]:
        """Fallback agent selection"""
        ext = file_type.lower()
        if any(x in ext for x in ['.jpg', '.png']):
            return ["vision"]
        elif any(x in ext for x in ['.mp4', '.avi']):
            return ["video", "vision"]
        elif any(x in ext for x in ['.mp3', '.wav']):
            return ["audio"]
        else:
            return ["vision"]

vertex_ai_manager = VertexAIManager()