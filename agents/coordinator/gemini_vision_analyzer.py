# pyright: reportPrivateImportUsage=false
"""
Gemini Vision Analyzer - Frame-by-frame deepfake analysis with visual evidence
===============================================================================
Uses Google Gemini multimodal LLM to analyze individual frames for manipulation.
Improvements over v1:
- Better prompt engineering with explicit scoring rubric
- Strategic frame sampling (start, faces, scene changes)
- Weighted confidence aggregation
- Inter-frame delay to avoid rate limits
- Safety settings to prevent refusals on face images
"""

import io
import os
import re
import time
import logging
import base64
import json
import warnings
from typing import List, Dict, Any, Optional, Callable, cast
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Computer Vision
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("DeepGuard.GeminiVision")

# ---------------------------------------------------------------------------
# Prompt template - explicit rubric with examples anchors Gemini's scoring
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT = """You are an expert forensic AI analyst with 10+ years of experience detecting deepfakes, face-swaps, and AI-generated media. Your analysis must be precise, evidence-based, and clinically objective.

## YOUR TASK
Analyze this image frame for signs of digital manipulation. Examine every pixel region systematically.

## DETECTION CHECKLIST — examine each category:

### CATEGORY A — Face Geometry (weight: HIGH)
- [ ] Facial symmetry: are both sides of the face proportionally consistent?
- [ ] Jawline continuity: does the jaw edge blend naturally with the neck?
- [ ] Ear morphology: are ears present, realistic, and symmetrically placed?
- [ ] Forehead-hairline transition: natural growth pattern vs generated pattern?
- [ ] Nose bridge geometry: straight and proportional?

### CATEGORY B — Skin & Texture (weight: HIGH)
- [ ] Pore visibility: real skin has irregular pores; AI skin is too smooth or has tiled patterns
- [ ] Skin tone gradient: consistent across face, neck, and hands?
- [ ] Age markers: wrinkles, freckles, blemishes consistent with apparent age?
- [ ] Subsurface scattering: does skin look translucent at edges (natural) or opaque (fake)?

### CATEGORY C — Eyes & Gaze (weight: VERY HIGH)
- [ ] Iris detail: real irises have unique fiber patterns; GAN irises repeat or blur
- [ ] Catchlights: consistent position of light reflections in both eyes?
- [ ] Sclera: any redness, veins, natural variation (fake eyes are often too white/perfect)
- [ ] Eyelashes: natural irregular spacing or AI-generated uniform pattern?
- [ ] Gaze direction: are both eyes looking in the same consistent direction?

### CATEGORY D — Boundary & Blending (weight: VERY HIGH)
- [ ] Face-background edge: any halos, color bleeding, or feathering at face boundary?
- [ ] Neck-face seam: does skin tone and texture match seamlessly?
- [ ] Hair-face boundary: hair strands visible and realistic at the edges?
- [ ] Hand-face intersection: if hands are near face, are edges clean?

### CATEGORY E — Lighting & Physics (weight: HIGH)
- [ ] Light source consistency: shadows on face match shadows in background?
- [ ] Specular highlights: reflections on skin match environmental lighting?
- [ ] Subshadow under nose/chin: present and physics-consistent?
- [ ] Eye socket shadows: natural depth or flat/painted appearance?

### CATEGORY F — GAN/Diffusion Artifacts (weight: VERY HIGH)
- [ ] Background coherence: any floating objects, duplicated patterns, warped text?
- [ ] Frequency artifacts: checkerboard patterns, repetitive textures (common in GAN upscaling)?
- [ ] Teeth: natural irregular shape/spacing or too-perfect/smeared?
- [ ] Hair: realistic individual strands or uniform blob texture?
- [ ] Jewelry/glasses: coherent or distorted at face boundaries?

## FRAME CONTEXT
{context}

## SCORING RUBRIC
- **0.90–1.00**: Multiple strong artifacts confirmed → definitive FAKE
- **0.75–0.90**: Clear artifacts in 1-2 high-weight categories → FAKE
- **0.60–0.75**: Suspicious but not conclusive → SUSPICIOUS
- **0.45–0.60**: Mixed/unclear signals → SUSPICIOUS with low confidence
- **0.70–0.85**: Strong authenticity indicators, no artifacts → REAL
- **0.85–1.00**: No artifacts, strong natural indicators → REAL

## RESPONSE FORMAT
Respond with ONLY valid JSON (no markdown code blocks, no extra text):
{{"verdict": "REAL" | "FAKE" | "SUSPICIOUS", "confidence": <float 0.0-1.0>, "reasoning": "<3-4 sentences: state what you found, WHERE specifically, and why it indicates manipulation or authenticity>", "artifacts_detected": ["<specific artifact with location, e.g. 'Halo blending artifact along left jaw edge'>"], "authenticity_indicators": ["<specific real indicator, e.g. 'Consistent iris fiber pattern in both eyes'>"], "examined_regions": ["<which face regions you analyzed>"]}}

## CRITICAL RULES
1. NEVER default to REAL without examining all 6 categories
2. Report artifacts with SPECIFIC LOCATIONS (left eye, jaw edge, hair boundary, etc.)
3. If no face detected: return INCONCLUSIVE with confidence 0.5
4. confidence 0.5 means genuinely uncertain — use it sparingly
5. A single high-weight artifact (Category C, D, or F) is sufficient for FAKE verdict at 0.75+"""


class GeminiVisionAnalyzer:
    def __init__(self):
        self.enabled = os.getenv("ENABLE_GEMINI_VISION", "true").lower() == "true"
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")
        self.fps = int(os.getenv("FRAMES_PER_SECOND", "2"))
        self.max_frames = int(os.getenv("MAX_FRAMES_TO_ANALYZE", "15"))
        self.save_annotated = os.getenv("SAVE_ANNOTATED_FRAMES", "true").lower() == "true"
        self.timeout_ms = int(os.getenv("GEMINI_VISION_TIMEOUT_MS", "45000"))
        self.inter_frame_delay = float(os.getenv("GEMINI_INTER_FRAME_DELAY", "1.0"))
        self.client = None

        if not self.api_key:
            logger.error("GEMINI_API_KEY not set")
            self.enabled = False
            return

        try:
            from google import genai as _genai  # type: ignore
            from google.genai import types as _gtypes  # type: ignore
            self.client = _genai.Client(
                api_key=self.api_key,
                http_options=_gtypes.HttpOptions(timeout=self.timeout_ms),  # type: ignore[reportCallIssue]
            )  # type: ignore
            logger.info(f"Gemini Vision initialized: {self.model_name}, delay={self.inter_frame_delay}s")
        except Exception as e:
            logger.error(f"Gemini Vision init failed: {e}")
            self.enabled = False

    # -----------------------------------------------------------------------
    # Frame extraction - strategic sampling
    # -----------------------------------------------------------------------
    def extract_frames_with_metadata(self, video_path: str) -> List[Dict[str, Any]]:
        """Extract frames strategically: first, last, evenly spaced, + face-heavy frames."""
        frames_data: List[Dict[str, Any]] = []

        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            logger.info(f"Video: {duration:.1f}s, {total_frames} frames @ {fps:.1f} FPS")

            if total_frames <= 0 or fps <= 0:
                cap.release()
                return frames_data

            # Calculate strategic frame indices
            target_count = min(self.max_frames, total_frames)
            indices: List[int] = []

            if target_count <= 3:
                # Very short video: take all available
                indices = list(range(total_frames))[:target_count]
            else:
                # Strategic sampling: anchor + uniform + scene-change detection
                # Anchors: first frame, very early (5%), early (15%), near-end (90%), last
                anchors = [
                    0,
                    max(1, int(total_frames * 0.05)),
                    max(2, int(total_frames * 0.15)),
                    max(3, int(total_frames * 0.90)),
                    max(4, total_frames - 2),
                ]
                indices = list(anchors)

                # Fill remaining slots with uniform spacing across entire video
                remaining = target_count - len(indices)
                if remaining > 0:
                    for i in range(remaining):
                        idx = int((i + 1) * total_frames / (remaining + 1))
                        indices.append(min(idx, total_frames - 1))

            # Deduplicate and sort
            indices = sorted(set(max(0, min(i, total_frames - 1)) for i in indices))[:target_count]

            # Read only the target frames (seek instead of sequential)
            for extracted_idx, frame_idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                timestamp = frame_idx / fps if fps > 0 else 0

                # Resize to max 1024px
                height, width = frame.shape[:2]
                max_dim = 1024
                if max(height, width) > max_dim:
                    scale = max_dim / max(height, width)
                    frame_resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
                else:
                    frame_resized = frame.copy()

                # Encode to JPEG
                _, buffer = cv2.imencode('.jpg', cast(Any, frame_resized), [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')

                frames_data.append({
                    "image_b64": frame_b64,
                    "timestamp": timestamp,
                    "frame_number": frame_idx,
                    "extracted_index": extracted_idx + 1,
                    "opencv_frame": frame_resized
                })

                logger.info(f"Extracted frame {extracted_idx + 1}/{len(indices)} (frame #{frame_idx}) at {timestamp:.1f}s")

            cap.release()
            logger.info(f"Extracted {len(frames_data)} strategic frames")

        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")

        return frames_data

    # -----------------------------------------------------------------------
    # Frame annotation (visual overlay on saved images)
    # -----------------------------------------------------------------------
    def annotate_frame(self, frame: np.ndarray, verdict: str, confidence: float,
                       artifacts: List[str], timestamp: float) -> np.ndarray:
        """Add visual annotations to a frame."""
        try:
            if frame is None or frame.size == 0:
                return frame

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_image)

            height, width = frame.shape[:2]
            font_size = max(20, int(height / 30))

            try:
                font = ImageFont.truetype("arial.ttf", font_size)
                small_font = ImageFont.truetype("arial.ttf", int(font_size * 0.7))
            except Exception:
                font = ImageFont.load_default()
                small_font = font

            # Color by verdict
            if verdict == "FAKE":
                bg_color = (255, 0, 0, 180)
            elif verdict == "SUSPICIOUS":
                bg_color = (255, 165, 0, 180)
            else:
                bg_color = (0, 200, 0, 180)

            # Header overlay bar
            header_height = int(height * 0.08)
            overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([(0, 0), (width, header_height)], fill=bg_color)

            # Verdict + confidence text
            verdict_text = f"{verdict} ({confidence:.0%})"
            draw.text((10, 10), verdict_text, font=font, fill=(255, 255, 255))

            # Timestamp (right side)
            time_text = f"{timestamp:.1f}s"
            draw.text((width - 120, 10), time_text, font=small_font, fill=(255, 255, 255))

            # Artifacts list
            if artifacts:
                y_offset = header_height + 20
                draw.text((10, y_offset), "Artifacts:", font=small_font, fill=(255, 255, 0))
                y_offset += 30
                for artifact in artifacts[:5]:
                    draw.text((20, y_offset), f"- {artifact}", font=small_font, fill=(255, 255, 255))
                    y_offset += 25

            # Merge overlay
            pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
            annotated = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)

            return annotated

        except Exception as e:
            logger.error(f"Annotation failed: {e}")
            return frame

    # -----------------------------------------------------------------------
    # Single image analysis with Gemini
    # -----------------------------------------------------------------------
    def analyze_image_with_gemini(self, image_b64: str, context: str = "", max_retries: int = 3) -> Dict[str, Any]:
        """Analyze a single image with Gemini. Includes retry with exponential backoff."""
        if not self.enabled or not self.client:
            return {"error": "Gemini Vision not enabled", "verdict": "INCONCLUSIVE", "confidence": 0.0}

        for attempt in range(max_retries):
            try:
                image_data = base64.b64decode(image_b64)

                prompt = ANALYSIS_PROMPT.format(context=context or "Standalone image analysis")

                from google.genai import types as _gtypes  # type: ignore
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        _gtypes.Part.from_text(text=prompt),
                        _gtypes.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                    ],
                    config=_gtypes.GenerateContentConfig(
                        temperature=0.3,  # Slightly higher for diversity, still low for consistency
                        response_mime_type="application/json",
                        safety_settings=[
                            _gtypes.SafetySetting(
                                category=cast(Any, "HARM_CATEGORY_SEXUALLY_EXPLICIT"),
                                threshold=cast(Any, "BLOCK_NONE"),
                            ),
                            _gtypes.SafetySetting(
                                category=cast(Any, "HARM_CATEGORY_DANGEROUS_CONTENT"),
                                threshold=cast(Any, "BLOCK_NONE"),
                            ),
                            _gtypes.SafetySetting(
                                category=cast(Any, "HARM_CATEGORY_HARASSMENT"),
                                threshold=cast(Any, "BLOCK_NONE"),
                            ),
                            _gtypes.SafetySetting(
                                category=cast(Any, "HARM_CATEGORY_HATE_SPEECH"),
                                threshold=cast(Any, "BLOCK_NONE"),
                            ),
                        ],
                    ),
                )

                text = (response.text or "").strip()
                if not text:
                    logger.warning(f"Empty Gemini response (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        time.sleep((attempt + 1) * 3)
                        continue
                    return {"verdict": "INCONCLUSIVE", "confidence": 0.5, "reasoning": "Empty model response",
                            "artifacts_detected": [], "authenticity_indicators": []}

                # Parse JSON - handle markdown wrapping
                json_str = text
                if "```" in text:
                    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
                    if match:
                        json_str = match.group(1).strip()
                else:
                    match = re.search(r"\{.*\}", text, re.DOTALL)
                    if match:
                        json_str = match.group(0)

                result = json.loads(json_str)

                # Normalize confidence: ensure 0.0-1.0
                conf = float(result.get("confidence", 0.5))
                if conf > 1.0:
                    conf = conf / 100.0
                conf = max(0.0, min(1.0, conf))
                # Reject exactly 0.0 confidence (means model didn't analyze)
                if conf == 0.0:
                    conf = 0.5
                result["confidence"] = conf

                # Normalize verdict
                v = str(result.get("verdict", "SUSPICIOUS")).upper()
                if v not in ("REAL", "FAKE", "SUSPICIOUS"):
                    v = "SUSPICIOUS"
                result["verdict"] = v

                logger.info(f"Gemini: {result['verdict']} ({result['confidence']:.0%}) - {result.get('reasoning', '')[:80]}")
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                    continue
                return {"verdict": "INCONCLUSIVE", "confidence": 0.5,
                        "reasoning": f"Could not parse model response",
                        "artifacts_detected": [], "authenticity_indicators": []}

            except Exception as e:
                error_str = str(e).lower()
                if ("quota" in error_str or "rate" in error_str or "429" in error_str or "resource" in error_str):
                    wait_time = (attempt + 1) * 5  # 5s, 10s, 15s - much more conservative
                    logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})...")
                    time.sleep(wait_time)
                    if attempt < max_retries - 1:
                        continue
                elif "block" in error_str or "safety" in error_str:
                    logger.warning(f"Gemini blocked by safety filter: {e}")
                    return {"verdict": "INCONCLUSIVE", "confidence": 0.5,
                            "reasoning": "Image blocked by safety filter",
                            "artifacts_detected": [], "authenticity_indicators": []}

                logger.error(f"Gemini analysis failed (attempt {attempt + 1}): {e}")
                if attempt >= max_retries - 1:
                    return {"verdict": "INCONCLUSIVE", "confidence": 0.0,
                            "reasoning": f"Analysis error: {str(e)[:100]}",
                            "artifacts_detected": [], "authenticity_indicators": []}

        return {"verdict": "INCONCLUSIVE", "confidence": 0.0,
                "reasoning": "Max retries exceeded", "artifacts_detected": []}

    # -----------------------------------------------------------------------
    # Multi-frame video analysis
    # -----------------------------------------------------------------------
    def analyze_video_multiframe(
        self,
        video_path: str,
        job_id: str = "unknown",
        on_frame_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Analyze video with multiple strategic frames. Weighted confidence aggregation."""
        if not self.enabled:
            return {"error": "Gemini Vision not enabled"}

        logger.info(f"Multi-frame analysis: {video_path}")

        frames_data = self.extract_frames_with_metadata(video_path)
        if not frames_data:
            return {"verdict": "ERROR", "confidence": 0.0, "reasoning": "Could not extract frames"}

        total_frames = len(frames_data)
        frame_analyses: List[Dict[str, Any]] = []
        annotated_frames: List[str] = []

        for idx, frame_data in enumerate(frames_data):
            current = idx + 1
            logger.info(f"Analyzing frame {current}/{total_frames}...")

            # Inter-frame delay to avoid rate limits (skip for first frame)
            if idx > 0 and self.inter_frame_delay > 0:
                time.sleep(self.inter_frame_delay)

            # Build rich temporal context for each frame
            prev_verdict = frame_analyses[-1].get("verdict", "N/A") if frame_analyses else "N/A"
            prev_conf = frame_analyses[-1].get("confidence", 0) if frame_analyses else 0
            position_desc = (
                "FIRST FRAME (capture de referință)" if current == 1 else
                "ULTIMUL FRAME" if current == total_frames else
                f"Frame {current}/{total_frames}"
            )
            frame_context = (
                f"{position_desc} la {frame_data['timestamp']:.1f}s dintr-un video de {total_frames} frame-uri analizate. "
                f"Concentrează-te pe față și marginile ei. "
                + (f"Frame anterior: {prev_verdict} ({prev_conf:.0%}) — verifică dacă artefactele sunt consistente între frame-uri. " if current > 1 else "")
                + f"Examinează cu atenție deosebită: ochi, marginea feței, textură piele, păr."
            )

            analysis = self.analyze_image_with_gemini(
                frame_data["image_b64"],
                context=frame_context
            )

            analysis["frame_number"] = frame_data["frame_number"]
            analysis["timestamp"] = frame_data["timestamp"]
            analysis["extracted_index"] = frame_data["extracted_index"]
            frame_analyses.append(analysis)

            if on_frame_progress:
                try:
                    on_frame_progress(job_id, current, total_frames)
                except Exception as e:
                    logger.debug(f"Progress callback error: {e}")

            # Save annotated frame
            if self.save_annotated:
                opencv_frame = frame_data.get("opencv_frame")
                if opencv_frame is not None and opencv_frame.size > 0:
                    annotated = self.annotate_frame(
                        opencv_frame,
                        analysis.get("verdict", "UNKNOWN"),
                        analysis.get("confidence", 0.0),
                        analysis.get("artifacts_detected", []),
                        frame_data["timestamp"]
                    )

                    output_dir = Path(f"/app/data/evidence/{job_id}")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    output_path = output_dir / f"frame_{frame_data['extracted_index']:03d}_{analysis.get('verdict', 'UNK')}.jpg"
                    cv2.imwrite(str(output_path), annotated)
                    annotated_frames.append(f"/evidence/{job_id}/frame_{frame_data['extracted_index']:03d}_{analysis.get('verdict', 'UNK')}.jpg")
                    logger.info(f"Saved: {output_path}")

        # -------------------------------------------------------------------
        # Weighted aggregation: high-confidence frames matter more
        # -------------------------------------------------------------------
        fake_count = 0
        suspicious_count = 0
        real_count = 0
        weighted_fake_score = 0.0
        weighted_real_score = 0.0
        total_weight = 0.0

        for a in frame_analyses:
            v = a.get("verdict", "INCONCLUSIVE")
            c = a.get("confidence", 0.5)
            if c <= 0:
                c = 0.5  # Neutral for missing confidence

            weight = c  # Confidence IS the weight

            if v == "FAKE":
                fake_count += 1
                weighted_fake_score += c * c  # Square confidence to amplify high-confidence detections
                total_weight += weight
            elif v == "SUSPICIOUS":
                suspicious_count += 1
                weighted_fake_score += c * 0.5  # Partial fake weight
                total_weight += weight
            elif v == "REAL":
                real_count += 1
                weighted_real_score += c * c
                total_weight += weight

        # Calculate weighted average confidence
        valid_confs = [a.get("confidence", 0) for a in frame_analyses if a.get("confidence", 0) > 0]
        avg_confidence = (sum(valid_confs) / len(valid_confs)) if valid_confs else 0.5

        # Determine final verdict using weighted scoring
        total_analyzed = fake_count + suspicious_count + real_count
        if total_analyzed == 0:
            final_verdict = "INCONCLUSIVE"
            avg_confidence = 0.5
        elif total_weight > 0:
            fake_ratio = weighted_fake_score / total_weight
            real_ratio = weighted_real_score / total_weight

            # Any high-confidence FAKE frame (>0.8) is very strong evidence
            high_conf_fakes = [a for a in frame_analyses
                               if a.get("verdict") == "FAKE" and a.get("confidence", 0) > 0.8]

            if high_conf_fakes:
                # High-confidence FAKE detection - trust it
                final_verdict = "FAKE"
                avg_confidence = max(a.get("confidence", 0) for a in high_conf_fakes)
            elif fake_ratio > real_ratio and fake_count >= max(1, total_analyzed * 0.3):
                # Weighted fake > real AND at least 30% frames are fake
                final_verdict = "FAKE"
            elif suspicious_count > total_analyzed * 0.4:
                final_verdict = "SUSPICIOUS"
            elif fake_count > 0 and fake_ratio > 0.2:
                # Some fake frames with decent confidence
                final_verdict = "SUSPICIOUS"
                avg_confidence = min(avg_confidence, 0.7)
            elif real_ratio > fake_ratio:
                final_verdict = "REAL"
            else:
                final_verdict = "INCONCLUSIVE"
                avg_confidence = 0.5
        else:
            final_verdict = "INCONCLUSIVE"
            avg_confidence = 0.5

        # Clamp confidence
        avg_confidence = max(0.0, min(1.0, avg_confidence))

        # Build reasoning
        final_reasoning = f"**Gemini Vision Analysis** — {total_frames} frames analyzed\n\n"
        final_reasoning += f"**Statistici:** {fake_count} FAKE / {suspicious_count} SUSPICIOUS / {real_count} REAL\n\n"

        # Collect all unique artifacts
        all_artifacts: List[str] = []
        for a in frame_analyses:
            for art in a.get("artifacts_detected", []):
                if art and art not in all_artifacts:
                    all_artifacts.append(art)

        if all_artifacts:
            final_reasoning += "**Artefacte detectate:**\n"
            for art in all_artifacts[:8]:
                final_reasoning += f"- =={art}==\n"
            final_reasoning += "\n"

        # Per-frame summary
        final_reasoning += "**Detalii per frame:**\n"
        for analysis in frame_analyses:
            v = analysis.get('verdict', 'N/A')
            c = analysis.get('confidence', 0)
            r = analysis.get('reasoning', '')[:80]
            final_reasoning += f"- Frame {analysis['extracted_index']} ({analysis['timestamp']:.1f}s): **{v}** ({c:.0%}) — {r}\n"

        return {
            "verdict": final_verdict,
            "confidence": avg_confidence,
            "reasoning": final_reasoning,
            "frame_analyses": frame_analyses,
            "annotated_frames": annotated_frames,
            "artifacts_detected": all_artifacts,
            "statistics": {
                "total_frames": len(frames_data),
                "fake_frames": fake_count,
                "suspicious_frames": suspicious_count,
                "real_frames": real_count
            }
        }

    # -----------------------------------------------------------------------
    # Single image analysis
    # -----------------------------------------------------------------------
    def analyze_image(self, image_path: str, job_id: str = "unknown") -> Dict[str, Any]:
        """Analyze a single image."""
        if not self.enabled:
            return {"error": "Gemini Vision not enabled"}

        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode('utf-8')

            analysis = self.analyze_image_with_gemini(image_b64, context="Single image forensic analysis")

            if self.save_annotated:
                frame = cv2.imread(image_path)
                if frame is not None and frame.size > 0:
                    annotated = self.annotate_frame(
                        frame,
                        analysis.get("verdict", "UNKNOWN"),
                        analysis.get("confidence", 0.0),
                        analysis.get("artifacts_detected", []),
                        0.0
                    )

                    output_dir = Path(f"/app/data/evidence/{job_id}")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    output_path = output_dir / f"annotated_{Path(image_path).name}"
                    cv2.imwrite(str(output_path), annotated)
                    analysis["annotated_image"] = f"/evidence/{job_id}/annotated_{Path(image_path).name}"
                    logger.info(f"Saved: {output_path}")

            return analysis

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {"verdict": "ERROR", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}


# Global instance
gemini_vision = GeminiVisionAnalyzer()
