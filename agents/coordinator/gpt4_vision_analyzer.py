# pyright: reportPrivateImportUsage=false
"""
GPT-4o Vision Analyzer - Frame-by-frame deepfake analysis using OpenAI GPT-4o
===============================================================================
Drop-in replacement for GeminiVisionAnalyzer:
- Same public API: analyze_video_multiframe(), analyze_image()
- Inherits: extract_frames_with_metadata(), annotate_frame(), aggregation logic
- Overrides: analyze_image_with_gemini() → analyze_image_with_gpt4()

Requires:
- OPENAI_API_KEY env var
- ENABLE_GPT4_VISION=true
- GPT4_VISION_MODEL=gpt-4o (default)
"""

import os
import re
import time
import logging
import base64
import json
from typing import List, Dict, Any, Optional, Callable

from gemini_vision_analyzer import GeminiVisionAnalyzer

logger = logging.getLogger("DeepGuard.GPT4Vision")

# Compact prompt for GPT-4o (~220 tokens vs ~1400 for the Gemini prompt).
# Keeping it short is important: with detail="low" image tokens (~85) + this prompt
# we stay well under 512 input tokens, leaving plenty of room for max_tokens=2048 output.
GPT4_ANALYSIS_PROMPT = """Analyze this image for deepfake/AI manipulation. Return ONLY a JSON object with exactly these fields:
{{
  "verdict": "REAL" or "FAKE" or "SUSPICIOUS",
  "confidence": 0.0-1.0,
  "reasoning": "brief 1-2 sentence explanation",
  "artifacts_detected": ["list of manipulation artifacts found, or empty list"],
  "authenticity_indicators": ["list of authentic features found, or empty list"]
}}
Inspect two groups of signals:
(1) Face-SWAP tells: skin texture uniformity, facial boundary blending, hair/teeth rendering, lighting consistency, GAN frequency artifacts, eye reflections, ear/jawline geometry.
(2) Full-scene generative tells (for whole-frame AI video/image generators like Veo, Sora, Imagen, Midjourney, which have no blending seam at all): camera sensor noise across the WHOLE frame vs. uniformly "denoised" look, background text legibility, whole-scene shadow/reflection physics, object permanence.
IMPORTANT: absence of group-(1) artifacts does NOT prove authenticity — full-scene generators are expected to look clean there by design. Only mark REAL when you find POSITIVE group-(2) evidence (real sensor noise, correct scene-wide physics); otherwise prefer SUSPICIOUS over REAL. Context: {context}"""


class GPT4VisionAnalyzer(GeminiVisionAnalyzer):
    """
    GPT-4o Vision implementation.
    Inherits frame extraction, annotation, and aggregation from GeminiVisionAnalyzer.
    Only the actual model call is replaced.
    """

    def __init__(self):
        # Initialize parent WITHOUT enabling Gemini (we override the API call)
        # We skip parent __init__ and initialize manually to avoid Gemini client creation
        self.enabled = os.getenv("ENABLE_GPT4_VISION", "true").lower() == "true"
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("GPT4_VISION_MODEL", "gpt-4o")
        self.fps = int(os.getenv("FRAMES_PER_SECOND", "2"))
        self.max_frames = int(os.getenv("MAX_FRAMES_TO_ANALYZE", "15"))
        self.save_annotated = os.getenv("SAVE_ANNOTATED_FRAMES", "true").lower() == "true"
        self.timeout_ms = int(os.getenv("GPT4_VISION_TIMEOUT_MS", "60000"))
        # Default 2.0s — conservative to avoid rate limits on Pay-as-you-go tier
        self.inter_frame_delay = float(os.getenv("GPT4_INTER_FRAME_DELAY", "2.0"))
        self.client = None  # Keep parent attribute for compat

        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY not set — GPT-4o Vision disabled")
            self.enabled = False
            return

        try:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=self.openai_api_key,
                timeout=self.timeout_ms / 1000,  # Convert ms → seconds
            )
            logger.info(f"GPT-4o Vision initialized: model={self.model_name}, delay={self.inter_frame_delay}s")
        except Exception as e:
            logger.error(f"GPT-4o Vision init failed: {e}")
            self.enabled = False

    # -----------------------------------------------------------------------
    # Override: single image analysis using OpenAI GPT-4o
    # -----------------------------------------------------------------------
    def analyze_image_with_gemini(self, image_b64: str, context: str = "", max_retries: int = 3) -> Dict[str, Any]:
        """
        Override parent method to use GPT-4o Vision instead of Gemini.
        Same signature and return format so the rest of the pipeline works unchanged.
        """
        return self.analyze_image_with_gpt4(image_b64, context, max_retries)

    def analyze_image_with_gpt4(self, image_b64: str, context: str = "", max_retries: int = 3) -> Dict[str, Any]:
        """Analyze a single image with GPT-4o Vision. Retries with exponential backoff."""
        if not self.enabled:
            return {"error": "GPT-4o Vision not enabled", "verdict": "INCONCLUSIVE", "confidence": 0.0}

        for attempt in range(max_retries):
            try:
                prompt_text = GPT4_ANALYSIS_PROMPT.format(context=context or "standalone frame")

                # GPT-4o accepts base64 images via data URI
                image_data_uri = f"data:image/jpeg;base64,{image_b64}"

                response = self._openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert forensic AI analyst specializing in deepfake detection. "
                                "You MUST respond with ONLY a valid JSON object. "
                                "Do NOT use markdown, code blocks, or any extra text. "
                                "Start your response directly with { and end with }."
                            )
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt_text
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_uri,
                                        "detail": "low"  # saves ~680 tokens vs "high"; sufficient for forensic analysis
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=2048,   # was 1024 — must be large enough for full JSON response
                    temperature=0.2,   # Low temperature for consistent forensic analysis
                    # NOTE: response_format={"type": "json_object"} is NOT used here.
                    # OpenAI's JSON mode is incompatible with vision/image inputs and
                    # causes empty responses. We rely on prompt engineering + parsing instead.
                )

                raw_text = (response.choices[0].message.content or "").strip()

                if not raw_text:
                    logger.warning(f"Empty GPT-4o response (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        time.sleep((attempt + 1) * 3)
                        continue
                    return {"verdict": "INCONCLUSIVE", "confidence": 0.5,
                            "reasoning": "Empty model response",
                            "artifacts_detected": [], "authenticity_indicators": []}

                # ------------------------------------------------------------------
                # Robust JSON parsing — 3 strategies
                # ------------------------------------------------------------------
                result = None

                # Strategy 1: direct parse (model returned clean JSON)
                try:
                    result = json.loads(raw_text)
                except json.JSONDecodeError:
                    pass

                # Strategy 2: strip markdown code block wrapper
                if result is None and "```" in raw_text:
                    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_text)
                    if match:
                        try:
                            result = json.loads(match.group(1).strip())
                        except json.JSONDecodeError:
                            pass

                # Strategy 3: extract from first { to last }
                if result is None:
                    start = raw_text.find("{")
                    end = raw_text.rfind("}") + 1
                    if start != -1 and end > start:
                        try:
                            result = json.loads(raw_text[start:end])
                        except json.JSONDecodeError:
                            pass

                if result is None:
                    logger.error(
                        f"JSON parse failed (attempt {attempt + 1}). "
                        f"Full raw response ({len(raw_text)} chars): {raw_text}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep((attempt + 1) * 2)
                        continue
                    return {"verdict": "INCONCLUSIVE", "confidence": 0.5,
                            "reasoning": "Could not parse model response",
                            "artifacts_detected": [], "authenticity_indicators": []}

                # Normalize confidence
                conf = float(result.get("confidence", 0.5))
                if conf > 1.0:
                    conf = conf / 100.0
                conf = max(0.0, min(1.0, conf))
                if conf == 0.0:
                    conf = 0.5
                result["confidence"] = conf

                # Normalize verdict
                v = str(result.get("verdict", "SUSPICIOUS")).upper()
                if v not in ("REAL", "FAKE", "SUSPICIOUS"):
                    v = "SUSPICIOUS"
                result["verdict"] = v

                # Ensure required fields exist
                result.setdefault("artifacts_detected", [])
                result.setdefault("authenticity_indicators", [])
                result.setdefault("reasoning", "")

                logger.info(
                    f"GPT-4o: {result['verdict']} ({result['confidence']:.0%}) "
                    f"— {result.get('reasoning', '')[:80]}"
                )
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                    continue
                return {"verdict": "INCONCLUSIVE", "confidence": 0.5,
                        "reasoning": "Could not parse model response",
                        "artifacts_detected": [], "authenticity_indicators": []}

            except Exception as e:
                error_str = str(e).lower()

                # Rate limit handling
                if any(k in error_str for k in ("rate_limit", "rate limit", "429", "too many requests")):
                    wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                    logger.warning(f"GPT-4o rate limited, waiting {wait_time}s (attempt {attempt + 1})...")
                    time.sleep(wait_time)
                    if attempt < max_retries - 1:
                        continue

                # Content policy / safety
                elif any(k in error_str for k in ("content_policy", "safety", "filtered", "refused")):
                    logger.warning(f"GPT-4o content policy block: {e}")
                    return {"verdict": "INCONCLUSIVE", "confidence": 0.5,
                            "reasoning": "Image blocked by content policy",
                            "artifacts_detected": [], "authenticity_indicators": []}

                # Invalid API key
                elif any(k in error_str for k in ("authentication", "invalid_api_key", "401", "unauthorized")):
                    logger.error(f"GPT-4o authentication failed: {e}")
                    self.enabled = False
                    return {"verdict": "INCONCLUSIVE", "confidence": 0.0,
                            "reasoning": "Authentication failed — check OPENAI_API_KEY",
                            "artifacts_detected": [], "authenticity_indicators": []}

                logger.error(f"GPT-4o analysis failed (attempt {attempt + 1}): {e}")
                if attempt >= max_retries - 1:
                    return {"verdict": "INCONCLUSIVE", "confidence": 0.0,
                            "reasoning": f"Analysis error: {str(e)[:100]}",
                            "artifacts_detected": [], "authenticity_indicators": []}

        return {"verdict": "INCONCLUSIVE", "confidence": 0.0,
                "reasoning": "Max retries exceeded", "artifacts_detected": []}

    # -----------------------------------------------------------------------
    # Override check: ensure 'enabled' reflects GPT-4o, not Gemini
    # -----------------------------------------------------------------------
    def analyze_video_multiframe(
        self,
        video_path: str,
        job_id: str = "unknown",
        on_frame_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Analyze video frame-by-frame using GPT-4o Vision."""
        if not self.enabled:
            return {"error": "GPT-4o Vision not enabled"}

        # Reuse parent implementation — it calls self.analyze_image_with_gemini()
        # which we've overridden to call GPT-4o instead
        logger.info(f"GPT-4o Vision multi-frame analysis: {video_path}")
        result = super().analyze_video_multiframe(video_path, job_id, on_frame_progress)

        # Update reasoning header to reflect GPT-4o
        if "reasoning" in result and isinstance(result["reasoning"], str):
            result["reasoning"] = result["reasoning"].replace(
                "**Gemini Vision Analysis**", "**GPT-4o Vision Analysis**"
            )

        return result

    def analyze_image(self, image_path: str, job_id: str = "unknown") -> Dict[str, Any]:
        """Analyze a single image using GPT-4o Vision."""
        if not self.enabled:
            return {"error": "GPT-4o Vision not enabled"}

        logger.info(f"GPT-4o Vision single image analysis: {image_path}")
        return super().analyze_image(image_path, job_id)


# ---------------------------------------------------------------------------
# Factory: returns GPT-4o if enabled + key present, else falls back to Gemini
# ---------------------------------------------------------------------------
def get_vision_analyzer():
    """
    Returns GPT-4o Vision analyzer if ENABLE_GPT4_VISION=true and OPENAI_API_KEY is set.
    Falls back to GeminiVisionAnalyzer otherwise.
    """
    enable_gpt4 = os.getenv("ENABLE_GPT4_VISION", "true").lower() == "true"
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if enable_gpt4 and openai_key:
        analyzer = GPT4VisionAnalyzer()
        if analyzer.enabled:
            logger.info("✅ Active vision analyzer: GPT-4o Vision")
            return analyzer
        else:
            logger.warning("GPT-4o Vision init failed — falling back to Gemini Vision")

    # Fallback
    from gemini_vision_analyzer import gemini_vision
    logger.info("✅ Active vision analyzer: Gemini Vision (fallback)")
    return gemini_vision


# Global instance — auto-selects based on env config
gpt4_vision = GPT4VisionAnalyzer()
active_vision_analyzer = get_vision_analyzer()
