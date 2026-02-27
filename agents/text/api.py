"""
DeepGuard Text Agent - AI-Generated Text Detection
==================================================
Detects LLM-generated text using RoBERTa classifier.
"""

import os
import time
import threading
import logging
from typing import Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepGuardText")

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="DeepGuard Text Agent",
    version="2.0",
    description="AI-generated text detection using RoBERTa"
)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/best_text_model.pth"
TOKENIZER_PATH = "./tokenizer_model"
BASE_MODEL = "roberta-base"

# Global model variables
model = None
tokenizer = None

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================
class TextRequest(BaseModel):
    """Request schema for text analysis"""
    text: str = Field(..., description="Text to analyze")

class TextResponse(BaseModel):
    """Response schema for text analysis"""
    status: str = Field(..., description="OK | ERROR | INCONCLUSIVE")
    verdict: str = Field(..., description="REAL | FAKE | INCONCLUSIVE_TECHNICAL | INCONCLUSIVE_INSUFFICIENT_DATA")
    fake_probability: float = Field(..., ge=0.0, le=100.0, description="Percentage probability of being AI-generated")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Model confidence score")
    details: str = Field(default="", description="Additional analysis details")
    reason: str = Field(default="", description="Explanation for status/verdict")

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.post("/analyze", response_model=TextResponse)
async def analyze_text(request: TextRequest) -> Dict[str, Any]:
    """
    Analyze text for AI-generated patterns.
    Standardized to never crash (no 500s).
    """
    # 1. Input Validation
    if not request.text or len(request.text.strip()) == 0:
        logger.warning("⚠️ Received empty text input")
        return {
            "status": "INCONCLUSIVE",
            "verdict": "INCONCLUSIVE_INSUFFICIENT_DATA",
            "fake_probability": 0.0,
            "confidence": 0.0,
            "details": "Input text is empty or whitespace only.",
            "reason": "Input data insufficient"
        }
    
    # 2. Model Availability Check
    if model is None or tokenizer is None:
        logger.error("❌ Model not loaded during request")
        return {
            "status": "ERROR",
            "verdict": "INCONCLUSIVE_TECHNICAL",
            "fake_probability": 0.0,
            "confidence": 0.0,
            "details": "Model or tokenizer not initialized.",
            "reason": "Service temporarily unavailable"
        }
    
    try:
        # 3. Tokenization & Inference
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            
            prob_real = probs[0][0].item() * 100
            prob_fake = probs[0][1].item() * 100  # Index 1 = FAKE (AI-generated)
        
        logger.info(f"📊 Analysis - Real: {prob_real:.2f}%, Fake: {prob_fake:.2f}%")
        
        # 4. Generate Standardized Response
        verdict = "FAKE" if prob_fake > 50 else "REAL"
        confidence = max(prob_real, prob_fake)
        
        return {
            "status": "OK",
            "verdict": verdict,
            "fake_probability": round(prob_fake, 2),
            "confidence": round(confidence, 2),
            "details": f"RoBERTa Classification: {prob_fake:.2f}% AI-generated probability",
            "reason": "Neural analysis completed successfully"
        }
    
    except Exception as e:
        logger.error(f"❌ Text analysis error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 5. Fail-Safe Return
        return {
            "status": "ERROR",
            "verdict": "INCONCLUSIVE_TECHNICAL",
            "fake_probability": 0.0,
            "confidence": 0.0,
            "details": f"Internal error: {str(e)[:100]}",
            "reason": "Agent crashed during inference"
        }

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for Docker"""
    return {
        "status": "healthy",
        "agent": "text",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": DEVICE,
        "mode": "neural" if model is not None else "loading"
    }

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with agent info"""
    return {
        "agent": "DeepGuard Text Agent",
        "version": "2.0",
        "model": BASE_MODEL,
        "status": "operational" if model is not None else "loading"
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def load_model_background():
    """Load model in background to avoid blocking startup"""
    global model, tokenizer
    try:
        logger.info(f"⏳ Loading model from {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH, 
            local_files_only=True,
            num_labels=2
        ).to(DEVICE)
        model.eval()
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

def register_with_coordinator():
    """Register this agent with the Brain Coordinator via MCP"""
    agent_config = {
        "name": "text",
        "display_name": "Text Agent",
        "description": "Detects LLM-generated text using RoBERTa classifier",
        "version": "2.0.0",
        "agent_type": "neural",
        "modality": "text",
        "capabilities": [
            {"name": "llm_detection", "description": "Identifies AI-generated text", "confidence_threshold": 0.8},
            {"name": "linguistic_analysis", "description": "Analyzes writing patterns", "confidence_threshold": 0.7}
        ],
        "input_schema": {
            "modality": "text",
            "formats": ["txt", "md", "json"],
            "max_size_mb": 10,
            "requires_preprocessing": False
        },
        "output_schema": {
            "verdict_field": "verdict",
            "confidence_field": "confidence",
            "details_field": "details"
        },
        "endpoint": "http://deepguard_text:8001/analyze",
        "health_endpoint": "http://deepguard_text:8001/health"
    }
    
    coordinator_url = os.getenv("COORDINATOR_URL", "http://deepguard_brain:8000")
    register_url = f"{coordinator_url}/mcp/register"
    max_retries = 15

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🔄 Attempting to register with Coordinator (attempt {attempt}/{max_retries})...")
            response = requests.post(register_url, json={"tool": agent_config}, timeout=5)
            if response.status_code in [200, 201]:
                logger.info("✅ Registered with Coordinator successfully!")
                return
            else:
                logger.error(f"⚠️ Registration failed: {response.text}")
        except Exception as e:
            logger.warning(f"⚠️ Coordinator unreachable ({e}). Retrying in 5s...")

        time.sleep(5)

    logger.error(f"❌ Failed to register with Coordinator after {max_retries} attempts")

# ============================================================================
# STARTUP EVENT
# ============================================================================
@app.on_event("startup")
async def startup_event() -> None:
    """
    Start server immediately, load model and register with MCP in background.
    """
    logger.info("🚀 Text Agent starting...")
    
    # Start model loading in background thread
    threading.Thread(target=load_model_background, daemon=True).start()
    
    # Start MCP registration in background thread
    threading.Thread(target=register_with_coordinator, daemon=True).start()
    
    logger.info("✅ Server started - model loading & MCP registration in progress...")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)