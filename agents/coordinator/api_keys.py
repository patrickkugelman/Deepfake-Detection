"""
API Key management for DeepGuard Pro SaaS.
Handles creation, validation, revocation, and usage tracking.
"""
import os
import secrets
import hashlib
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger("DeepGuard.APIKeys")

# Plan quotas: analyses per month
PLAN_QUOTAS = {
    "free":     100,
    "pro":      2000,
    "business": 10000,
    "enterprise": -1,  # unlimited
}

PLAN_PRICES = {
    "free":     0,
    "pro":      49,
    "business": 149,
}


def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key.
    Returns (raw_key, key_hash) — store only the hash, return raw once to user.
    """
    raw = "sk_live_" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    return raw, key_hash


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


# ============================================================================
# Neo4j operations
# ============================================================================

def create_api_key(neo4j_conn, username: str, name: str) -> Dict[str, Any]:
    """Create a new API key for a user. Returns the raw key (shown only once)."""
    raw_key, key_hash = generate_api_key()
    prefix = raw_key[:16]  # sk_live_XXXXXXXX — shown in listings

    neo4j_conn.query("""
        MATCH (u:User {username: $username})
        CREATE (k:APIKey {
            key_hash: $key_hash,
            prefix: $prefix,
            name: $name,
            created_at: datetime(),
            last_used: null,
            is_active: true,
            username: $username
        })
        CREATE (u)-[:HAS_KEY]->(k)
    """, {
        "username": username,
        "key_hash": key_hash,
        "prefix": prefix,
        "name": name,
    })

    logger.info(f"✅ API key created for {username}: {prefix}...")
    return {
        "key": raw_key,
        "prefix": prefix,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "message": "Save this key — it won't be shown again.",
    }


def list_api_keys(neo4j_conn, username: str):
    """List all API keys for a user (prefix only, never the full key)."""
    results = neo4j_conn.query("""
        MATCH (u:User {username: $username})-[:HAS_KEY]->(k:APIKey)
        RETURN k.prefix as prefix, k.name as name,
               k.created_at as created_at, k.last_used as last_used,
               k.is_active as is_active,
               id(k) as key_id
        ORDER BY k.created_at DESC
    """, {"username": username})
    return results or []


def revoke_api_key(neo4j_conn, username: str, prefix: str) -> bool:
    """Revoke an API key by prefix. Returns True if found and revoked."""
    result = neo4j_conn.query("""
        MATCH (u:User {username: $username})-[:HAS_KEY]->(k:APIKey {prefix: $prefix})
        SET k.is_active = false
        RETURN k.prefix as prefix
    """, {"username": username, "prefix": prefix})
    return bool(result)


def validate_api_key(neo4j_conn, raw_key: str) -> Optional[Dict[str, Any]]:
    """
    Validate an API key and return the associated user.
    Updates last_used timestamp.
    Returns None if invalid or revoked.
    """
    if not raw_key or not raw_key.startswith("sk_live_"):
        return None

    key_hash = hash_key(raw_key)

    results = neo4j_conn.query("""
        MATCH (u:User)-[:HAS_KEY]->(k:APIKey {key_hash: $key_hash, is_active: true})
        SET k.last_used = datetime()
        RETURN u.username as username, u.email as email, u.role as role,
               u.plan as plan, u.monthly_usage as monthly_usage,
               u.plan_expires_at as plan_expires_at,
               u.stripe_customer_id as stripe_customer_id
    """, {"key_hash": key_hash})

    if not results:
        return None

    user = dict(results[0])
    user["auth_method"] = "api_key"
    return user


# ============================================================================
# Usage tracking & quota enforcement
# ============================================================================

def get_user_plan(neo4j_conn, username: str) -> str:
    """Get user's current plan."""
    results = neo4j_conn.query(
        "MATCH (u:User {username: $u}) RETURN u.plan as plan",
        {"u": username}
    )
    if results and results[0].get("plan"):
        return str(results[0]["plan"])
    return "free"


def get_monthly_usage(neo4j_conn, username: str) -> int:
    """Get number of analyses this month."""
    results = neo4j_conn.query("""
        MATCH (u:User {username: $u})
        RETURN coalesce(u.monthly_usage, 0) as usage,
               coalesce(u.usage_reset_month, 0) as reset_month
    """, {"u": username})

    if not results:
        return 0

    row = results[0]
    current_month = datetime.now().month
    reset_month = int(row.get("reset_month") or 0)

    if reset_month != current_month:
        # New month — reset counter
        neo4j_conn.query("""
            MATCH (u:User {username: $u})
            SET u.monthly_usage = 0, u.usage_reset_month = $month
        """, {"u": username, "month": current_month})
        return 0

    return int(row.get("usage") or 0)


def increment_usage(neo4j_conn, username: str):
    """Increment monthly analysis counter for user."""
    current_month = datetime.now().month
    neo4j_conn.query("""
        MATCH (u:User {username: $u})
        SET u.monthly_usage = coalesce(u.monthly_usage, 0) + 1,
            u.usage_reset_month = $month,
            u.total_analyses = coalesce(u.total_analyses, 0) + 1
    """, {"u": username, "month": current_month})


def check_quota(neo4j_conn, username: str) -> Dict[str, Any]:
    """
    Check if user has remaining quota.
    Returns {"allowed": bool, "used": int, "limit": int, "plan": str}
    """
    plan = get_user_plan(neo4j_conn, username)
    limit = PLAN_QUOTAS.get(plan, PLAN_QUOTAS["free"])
    used = get_monthly_usage(neo4j_conn, username)

    if limit == -1:  # enterprise = unlimited
        return {"allowed": True, "used": used, "limit": -1, "plan": plan}

    return {
        "allowed": used < limit,
        "used": used,
        "limit": limit,
        "plan": plan,
        "remaining": max(0, limit - used),
    }


def ensure_user_plan_field(neo4j_conn, username: str):
    """Make sure user has plan field (migration helper for existing users)."""
    neo4j_conn.query("""
        MATCH (u:User {username: $u})
        WHERE u.plan IS NULL
        SET u.plan = 'free', u.monthly_usage = 0, u.usage_reset_month = 0
    """, {"u": username})
