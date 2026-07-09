"""
DeepGuard Pro — Slack Bot
Handles slash commands and app mentions. Responds async via response_url.
"""
import os
import json
import hmac
import hashlib
import time
import logging
import httpx
from typing import Optional

logger = logging.getLogger("DeepGuard.SlackBot")

SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")


# ============================================================================
# Signature verification
# ============================================================================

def verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    """Verify that the request came from Slack."""
    if not SLACK_SIGNING_SECRET:
        logger.warning("SLACK_SIGNING_SECRET not set — skipping signature check")
        return True

    # Reject stale requests (>5 minutes old)
    if abs(time.time() - int(timestamp)) > 300:
        return False

    sig_base = f"v0:{timestamp}:{body.decode()}"
    mac = hmac.new(SLACK_SIGNING_SECRET.encode(), sig_base.encode(), hashlib.sha256)
    expected = "v0=" + mac.hexdigest()
    return hmac.compare_digest(expected, signature)


# ============================================================================
# Message formatting (Block Kit)
# ============================================================================

def verdict_emoji(verdict: str) -> str:
    return {"FAKE": "🔴", "REAL": "🟢", "SUSPICIOUS": "🟡", "INCONCLUSIVE": "⚪"}.get(verdict.upper(), "⚪")


def format_result_blocks(job_result: dict) -> list:
    """Format analysis result as Slack Block Kit blocks."""
    verdict = (job_result.get("verdict") or job_result.get("final_verdict") or "INCONCLUSIVE").upper()
    confidence = job_result.get("confidence", 0)
    if isinstance(confidence, float) and confidence <= 1:
        confidence_pct = f"{confidence * 100:.1f}%"
    else:
        confidence_pct = f"{confidence}%"

    reasoning = job_result.get("reasoning") or "Analysis complete."
    filename = job_result.get("filename", "uploaded media")
    emoji = verdict_emoji(verdict)

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"DeepGuard Analysis — {emoji} {verdict}"}
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Verdict:* {emoji} `{verdict}`"},
                {"type": "mrkdwn", "text": f"*Confidence:* `{confidence_pct}`"},
                {"type": "mrkdwn", "text": f"*File:* `{filename}`"},
                {"type": "mrkdwn", "text": f"*Status:* Analysis complete"},
            ]
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Summary:*\n{reasoning[:500]}"}
        },
    ]

    # Show per-agent breakdown if available
    result_data = job_result.get("result") or {}
    if isinstance(result_data, str):
        try:
            result_data = json.loads(result_data)
        except Exception:
            result_data = {}

    agents = result_data.get("details") or result_data.get("agents") or {}
    if agents:
        agent_lines = []
        for name, data in agents.items():
            if isinstance(data, dict) and data.get("verdict"):
                v = data["verdict"].upper()
                c = data.get("confidence", "")
                c_str = f" ({c*100:.0f}%)" if isinstance(c, float) else ""
                agent_lines.append(f"• *{name}*: {verdict_emoji(v)} {v}{c_str}")
        if agent_lines:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Agent breakdown:*\n" + "\n".join(agent_lines[:6])}
            })

    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "Powered by DeepGuard Pro · deepguard.pro"}]
    })

    return blocks


def format_queued_blocks(job_id: str, source: str) -> list:
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"🔍 *Analyzing* `{source}`...\nThis takes 15–30 seconds. I'll post the result here."}
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"Job ID: `{job_id}`"}]
        }
    ]


def format_error_blocks(message: str) -> list:
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"❌ *Error:* {message}"}
        }
    ]


# ============================================================================
# Async result poster (called after Celery job completes)
# ============================================================================

async def post_result_to_slack(response_url: str, job_result: dict):
    """POST the analysis result back to Slack via response_url."""
    blocks = format_result_blocks(job_result)
    payload = {
        "response_type": "in_channel",
        "blocks": blocks,
        "replace_original": False,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(response_url, json=payload)
            if r.status_code != 200:
                logger.error(f"Slack response_url failed: {r.status_code} {r.text}")
    except Exception as e:
        logger.error(f"Failed to post to Slack response_url: {e}")


async def post_error_to_slack(response_url: str, message: str):
    payload = {
        "response_type": "ephemeral",
        "blocks": format_error_blocks(message),
        "replace_original": False,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(response_url, json=payload)
    except Exception as e:
        logger.error(f"Failed to post error to Slack: {e}")
