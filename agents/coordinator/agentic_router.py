"""
Agentic MCP Router - LLM-based agent selection from registry.
Uses MCP registry for candidates and GPT-4o for reasoning; fallback is deterministic per modality.
"""

import os
import re
import json
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("DeepGuard.AgenticRouter")

# Fallback when LLM is unavailable or returns invalid response
FALLBACK_VIDEO = ["video", "bio"]
FALLBACK_IMAGE = ["vision", "bio"]
FALLBACK_AUDIO = ["audio"]
FALLBACK_TEXT = ["text"]


def _extension_from_input(file_type: str, filename: str) -> str:
    """Derive file extension for registry lookup (e.g. 'mp4', 'jpg')."""
    if not file_type and not filename:
        return ""
    # Prefer extension from filename
    if filename and "." in filename:
        return filename.rsplit(".", 1)[-1].lower()
    # From file_type: ".mp4" -> "mp4", "video/mp4" -> "mp4"
    raw = (file_type or "").strip().lstrip(".").lower()
    if "/" in raw:
        raw = raw.split("/")[-1]
    return raw


def _deterministic_fallback(file_type: str, filename: str) -> List[str]:
    """Return fallback agent list by modality when LLM is not used."""
    ext = (file_type or "").lower()
    name_lower = (filename or "").lower()
    combined = ext + " " + name_lower

    if any(x in combined for x in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", "image"]):
        return list(FALLBACK_IMAGE)
    if any(x in combined for x in [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", "video"]):
        return list(FALLBACK_VIDEO)
    if any(x in combined for x in [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", "audio"]):
        return list(FALLBACK_AUDIO)
    if any(x in combined for x in [".txt", ".md", ".json", ".doc", ".docx", "text"]):
        return list(FALLBACK_TEXT)
    return list(FALLBACK_VIDEO)


def select_agents(
    file_type: str,
    filename: str,
    user_query: Optional[str] = None,
) -> Tuple[List[str], str]:
    """
    Select agents to invoke using MCP registry + LLM reasoning.

    Returns:
        (list of agent names, reasoning string for logs/UI)
    """
    try:
        from mcp_protocol import get_mcp_registry
    except ImportError:
        logger.warning("⚠️ mcp_protocol not available, using fallback")
        agents = _deterministic_fallback(file_type, filename)
        return agents, "Fallback: registry unavailable"

    ext = _extension_from_input(file_type, filename)
    registry = get_mcp_registry()
    candidates = registry.get_tools_for_extension(ext) if ext else []

    if not candidates:
        agents = _deterministic_fallback(file_type, filename)
        return agents, "Fallback: no matching tools for extension"

    candidate_names = [t.name for t in candidates]
    # Build prompt payload: name, description, modality, formats
    tools_desc = []
    for t in candidates:
        modality = getattr(t.input_schema, "modality", None)
        mod_str = str(modality) if modality else "unknown"
        formats = getattr(t.input_schema, "formats", []) or []
        tools_desc.append({
            "name": t.name,
            "description": t.description,
            "modality": mod_str,
            "formats": formats,
        })

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        agents = _deterministic_fallback(file_type, filename)
        return agents, "Fallback: OPENAI_API_KEY not set"

    prompt = _build_prompt(file_type, filename, user_query, tools_desc)
    try:
        from openai import OpenAI as _OpenAI  # type: ignore
        client = _OpenAI(api_key=openai_key, timeout=30.0)
        model_name = os.getenv("GPT4_VISION_MODEL", "gpt-4o")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI agent selector for a deepfake detection system. "
                        "Respond ONLY with valid JSON in this exact format: "
                        '{\"agents\": [\"name1\", \"name2\"], \"reasoning\": \"brief explanation\"}. '
                        "No markdown, no extra text."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.1,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("Empty LLM response")

        # Parse JSON (allow wrapped in markdown)
        json_str = text
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
        else:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                json_str = match.group(0)

        data = json.loads(json_str)
        agents_raw = data.get("agents")
        reasoning = data.get("reasoning", "")

        if not isinstance(agents_raw, list):
            raise ValueError("agents is not a list")

        # Validate: only allow agents that are in candidates
        agents = [a for a in agents_raw if isinstance(a, str) and a.strip() in candidate_names]
        if not agents:
            agents = _deterministic_fallback(file_type, filename)
            return agents, "Fallback: LLM returned no valid agents"

        return agents, reasoning or "Agentic routing"

    except Exception as e:
        logger.warning(f"⚠️ Agentic router LLM failed: {e}")
        agents = _deterministic_fallback(file_type, filename)
        return agents, f"Fallback: {e!s}"


def _build_prompt(
    file_type: str,
    filename: str,
    user_query: Optional[str],
    tools_desc: List[dict],
) -> str:
    tools_text = "\n".join(
        f"- {t['name']}: {t['description']} (modality: {t['modality']}, formats: {t['formats']})"
        for t in tools_desc
    )
    query_line = f"\nUser query (optional): {user_query}" if user_query else ""

    return f"""You are selecting which deepfake-detection agents to run for a single file.

Task: deepfake / AI-manipulation detection.
File: filename="{filename}", file_type="{file_type}"{query_line}

Available agents (candidates) that support this file type:
{tools_text}

We have 4 trained agents: vision (images), video (temporal video), audio (voice), text (text). Bio does rPPG pulse + liveness on video, and a separate skin-tone/texture/symmetry biological-plausibility check on still images (no pulse on images - a single frame carries no temporal signal). Choose from the candidates above the agents that make sense for this file. You may select one or more.

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{"agents": ["name1", "name2"], "reasoning": "brief explanation"}}
"""
