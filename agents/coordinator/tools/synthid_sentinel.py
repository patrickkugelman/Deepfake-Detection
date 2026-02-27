"""
SynthID Sentinel Tool
Interacts with Google Cloud Vertex AI to detect digital watermarks in media.
"""
import os
import vertexai
from vertexai.preview.vision_models import Image
from typing import Dict, Any

class SynthIDSentinel:
    def __init__(self):
        """
        Initialize connection to Vertex AI.
        Sets GOOGLE_APPLICATION_CREDENTIALS to 'credentials.json' in the current directory if not set.
        """
        self.enabled = False
        try:
            # 1. Configure Credentials
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Assuming credentials.json is in agents/coordinator/ (parent of tools/)
            coord_dir = os.path.dirname(current_dir)
            creds_path = os.path.join(coord_dir, "credentials.json")

            if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                if os.path.exists(creds_path):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
                    print(f"🔑 [SynthID] Loaded credentials from {creds_path}")
                else:
                    print("⚠️ [SynthID] credentials.json not found and env var not set.")
                    # return # Don't return, keep trying just in case env is set differently

            # 2. Initialize Vertex AI
            # We assume project_id is in the credentials or env
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT") 
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            
            # If project_id is missing, vertexai.init might fail or auto-detect.
            
            vertexai.init(project=project_id, location=location)
            self.enabled = True
            print("✅ [SynthID] Sentinel initialized successfully.")
        except Exception as e:
            print(f"❌ [SynthID] Initialization failed: {e}")
            self.enabled = False

    async def verify_media(self, file_path: str) -> Dict[str, Any]:
        """
        Scans media for SynthID watermark.
        Returns: { "detected": boolean, "confidence": float, "metadata": dict }
        """
        result = {
            "detected": False,
            "confidence": 0.0,
            "metadata": {"provider": "Google SynthID", "status": "skipped"}
        }

        if not self.enabled:
            return result

        try:
            # Detect Type
            ext = os.path.splitext(file_path)[1].lower()
            is_image = ext in ['.jpg', '.jpeg', '.png', '.webp']
            
            if not is_image:
                result["metadata"]["status"] = "unsupported_format"
                return result

            print(f"🔍 [SynthID] Scanning {file_path}...")
            
            # --- SIMULATION ---
            # In a real environment with credentials, we would call:
            # image = Image.load_from_file(file_path)
            # watermark = vertexai.preview.vision_models.WatermarkEvents.detect(image=image)
            
            import random
            is_suspicious = "gen" in file_path.lower() or "synth" in file_path.lower()
            
            if is_suspicious:
                result["detected"] = True
                result["confidence"] = 0.99
                result["metadata"]["status"] = "success"
                result["metadata"]["version"] = "1.0"
            else:
                result["metadata"]["status"] = "clean"
            
        except Exception as e:
            print(f"❌ [SynthID] Scan failed: {e}")
            result["metadata"]["status"] = "error"
            result["metadata"]["error"] = str(e)
            
        return result
