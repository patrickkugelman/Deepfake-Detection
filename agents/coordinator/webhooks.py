"""
Outbound webhooks for DeepGuard Pro SaaS.

Allows API clients to register a URL that will receive analysis results
automatically (push) instead of polling GET /jobs/{id}.

Payload is signed with HMAC-SHA256 using a per-webhook secret so the
receiver can verify authenticity.
"""
import os
import secrets
import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger("DeepGuard.Webhooks")

WEBHOOK_SIGNATURE_PREFIX = os.getenv("WEBHOOK_SIGNATURE_PREFIX", "sha256=")
WEBHOOK_MAX_FAILURES = int(os.getenv("WEBHOOK_MAX_FAILURES", "10"))
WEBHOOK_RETRY_DELAYS = [
    int(x) for x in os.getenv("WEBHOOK_RETRY_DELAYS", "60,300,900").split(",")
]

SUPPORTED_EVENTS = {
    "analysis.completed",
    "analysis.failed",
    "quota.exceeded",
}


# ============================================================================
# Secret generation / HMAC signing
# ============================================================================
def generate_webhook_secret() -> str:
    """whsec_ + 32 random url-safe chars."""
    return "whsec_" + secrets.token_urlsafe(32)


def sign_payload(payload: bytes, secret: str) -> str:
    """HMAC-SHA256 hex digest, prefixed per webhook convention."""
    mac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256)
    return f"{WEBHOOK_SIGNATURE_PREFIX}{mac.hexdigest()}"


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Constant-time comparison against the expected HMAC signature."""
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(expected, signature)


# ============================================================================
# Neo4j CRUD (WebhookEndpoint node)
# ============================================================================
def create_webhook(
    neo4j_conn,
    username: str,
    url: str,
    events: List[str],
) -> Dict[str, Any]:
    """Create a webhook endpoint. Returns the secret ONCE."""
    unknown = [e for e in events if e not in SUPPORTED_EVENTS]
    if unknown:
        raise ValueError(f"Unsupported events: {unknown}. Supported: {sorted(SUPPORTED_EVENTS)}")

    webhook_id = str(uuid.uuid4())
    secret = generate_webhook_secret()
    secret_hash = hashlib.sha256(secret.encode()).hexdigest()

    neo4j_conn.query("""
        MATCH (u:User {username: $username})
        CREATE (w:WebhookEndpoint {
            webhook_id: $webhook_id,
            url: $url,
            events: $events,
            secret_hash: $secret_hash,
            is_active: true,
            created_at: datetime(),
            last_triggered_at: null,
            failure_count: 0,
            success_count: 0,
            username: $username
        })
        CREATE (u)-[:OWNS_WEBHOOK]->(w)
    """, {
        "username": username,
        "webhook_id": webhook_id,
        "url": url,
        "events": events,
        "secret_hash": secret_hash,
    })

    logger.info(f"🪝 Webhook created for {username}: {webhook_id} -> {url}")
    return {
        "webhook_id": webhook_id,
        "url": url,
        "events": events,
        "secret": secret,  # shown once
        "created_at": datetime.now().isoformat(),
        "warning": "Store this secret. It won't be shown again.",
    }


def list_webhooks(neo4j_conn, username: str) -> List[Dict[str, Any]]:
    """Return all webhooks for a user (no secrets)."""
    results = neo4j_conn.query("""
        MATCH (u:User {username: $username})-[:OWNS_WEBHOOK]->(w:WebhookEndpoint)
        RETURN w.webhook_id as webhook_id, w.url as url, w.events as events,
               w.is_active as is_active, w.created_at as created_at,
               w.last_triggered_at as last_triggered_at,
               w.failure_count as failure_count, w.success_count as success_count
        ORDER BY w.created_at DESC
    """, {"username": username})
    return [
        {
            "webhook_id": r["webhook_id"],
            "url": r["url"],
            "events": r.get("events") or [],
            "is_active": bool(r.get("is_active")),
            "created_at": str(r["created_at"]) if r.get("created_at") else None,
            "last_triggered_at": str(r["last_triggered_at"]) if r.get("last_triggered_at") else None,
            "failure_count": int(r.get("failure_count") or 0),
            "success_count": int(r.get("success_count") or 0),
        }
        for r in (results or [])
    ]


def delete_webhook(neo4j_conn, username: str, webhook_id: str) -> bool:
    """Delete a webhook permanently."""
    result = neo4j_conn.query("""
        MATCH (u:User {username: $u})-[:OWNS_WEBHOOK]->(w:WebhookEndpoint {webhook_id: $wid})
        DETACH DELETE w
        RETURN count(w) as deleted
    """, {"u": username, "wid": webhook_id})
    return bool(result and result[0].get("deleted", 0) > 0)


def get_webhook(neo4j_conn, webhook_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single webhook by id (includes secret_hash for signing)."""
    results = neo4j_conn.query("""
        MATCH (w:WebhookEndpoint {webhook_id: $wid})
        RETURN w.webhook_id as webhook_id, w.url as url, w.events as events,
               w.secret_hash as secret_hash, w.is_active as is_active,
               w.username as username, w.failure_count as failure_count
    """, {"wid": webhook_id})
    return dict(results[0]) if results else None


def get_user_webhooks_for_event(
    neo4j_conn, username: str, event: str
) -> List[Dict[str, Any]]:
    """Return all active webhooks that want this specific event."""
    results = neo4j_conn.query("""
        MATCH (u:User {username: $u})-[:OWNS_WEBHOOK]->(w:WebhookEndpoint)
        WHERE w.is_active = true AND $event IN w.events
        RETURN w.webhook_id as webhook_id, w.url as url, w.events as events
    """, {"u": username, "event": event})
    return [dict(r) for r in (results or [])]


def record_delivery_success(neo4j_conn, webhook_id: str):
    neo4j_conn.query("""
        MATCH (w:WebhookEndpoint {webhook_id: $wid})
        SET w.last_triggered_at = datetime(),
            w.success_count = coalesce(w.success_count, 0) + 1,
            w.failure_count = 0
    """, {"wid": webhook_id})


def record_delivery_failure(neo4j_conn, webhook_id: str) -> int:
    """Increment failure counter. Disable webhook if threshold exceeded. Returns new failure count."""
    results = neo4j_conn.query("""
        MATCH (w:WebhookEndpoint {webhook_id: $wid})
        SET w.failure_count = coalesce(w.failure_count, 0) + 1,
            w.last_triggered_at = datetime()
        RETURN w.failure_count as fc
    """, {"wid": webhook_id})
    fc = int(results[0].get("fc", 0)) if results else 0

    if fc >= WEBHOOK_MAX_FAILURES:
        neo4j_conn.query("""
            MATCH (w:WebhookEndpoint {webhook_id: $wid})
            SET w.is_active = false,
                w.disabled_at = datetime(),
                w.disabled_reason = 'too_many_failures'
        """, {"wid": webhook_id})
        logger.warning(f"🚫 Webhook {webhook_id} disabled after {fc} failures")

    return fc


# ============================================================================
# Resolve secret for signing (we only stored the hash — but we need the secret
# to sign outbound payloads. Solution: we store the full secret too, encrypted
# at rest in production. For now we store a secondary field.)
# ============================================================================
#
# Design note: the spec originally said "store only the hash". That works for
# INBOUND webhooks (Stripe) but not outbound — we need to sign with the secret
# ourselves. So we additionally store the secret, keeping the hash for audit.
# In production, use KMS/sealed-secrets to encrypt the secret column.
def _store_plain_secret(neo4j_conn, webhook_id: str, secret: str):
    neo4j_conn.query("""
        MATCH (w:WebhookEndpoint {webhook_id: $wid})
        SET w.secret = $secret
    """, {"wid": webhook_id, "secret": secret})


def _get_plain_secret(neo4j_conn, webhook_id: str) -> Optional[str]:
    results = neo4j_conn.query(
        "MATCH (w:WebhookEndpoint {webhook_id: $wid}) RETURN w.secret as s",
        {"wid": webhook_id},
    )
    return results[0].get("s") if results else None


# Patch create_webhook to also persist the secret
_original_create = create_webhook


def create_webhook(neo4j_conn, username: str, url: str, events: List[str]) -> Dict[str, Any]:
    result = _original_create(neo4j_conn, username, url, events)
    _store_plain_secret(neo4j_conn, result["webhook_id"], result["secret"])
    return result


# ============================================================================
# Build standard payloads
# ============================================================================
def build_analysis_completed_payload(job_result: Dict[str, Any]) -> Dict[str, Any]:
    """Standard envelope for analysis.completed events."""
    return {
        "event": "analysis.completed",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": {
            "job_id": job_result.get("job_id"),
            "status": job_result.get("status", "completed"),
            "verdict": job_result.get("verdict"),
            "confidence": job_result.get("confidence"),
            "reasoning": job_result.get("reasoning", ""),
            "file_type": job_result.get("file_type"),
            "filename": job_result.get("filename"),
            "details": job_result.get("details", {}),
        },
    }


def build_analysis_failed_payload(job_id: str, error: str) -> Dict[str, Any]:
    return {
        "event": "analysis.failed",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": {"job_id": job_id, "status": "failed", "error": error},
    }


def build_test_payload(webhook_id: str) -> Dict[str, Any]:
    return {
        "event": "webhook.test",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": {"webhook_id": webhook_id, "message": "Test delivery from DeepGuard Pro."},
    }
