"""
Redis-backed rate limiter with sliding window algorithm.
Produces standard rate-limit headers for HTTP responses.
"""
import os
import time
import logging
from typing import Tuple, Dict, Any, Optional
import redis

logger = logging.getLogger("DeepGuard.RateLimiter")

# ============================================================================
# Redis connection
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DB_RATELIMIT = int(os.getenv("REDIS_DB_RATELIMIT", "2"))

try:
    _redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=REDIS_DB_RATELIMIT,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    _redis_client.ping()
    logger.info(f"✅ Rate limiter Redis connected (db={REDIS_DB_RATELIMIT})")
except Exception as e:
    logger.warning(f"⚠️ Rate limiter Redis unavailable: {e} — falling back to in-memory")
    _redis_client = None

# Fallback in-memory store (only used if Redis is down)
from collections import defaultdict
_memory_store: Dict[str, list] = defaultdict(list)


# ============================================================================
# Rate limit tiers
# ============================================================================
LIMITS = {
    "chat_analyze_auth":    (60, 60),    # 60 req / 60s for authenticated
    "chat_analyze_unauth":  (10, 60),    # 10 req / 60s unauthenticated
    "analyze_url_auth":     (60, 60),
    "auth_login":           (10, 60),
    "auth_register":        (5, 60),
    "api_keys_create":      (10, 60),
    "default":              (120, 60),
}


# ============================================================================
# Sliding window rate limiter
# ============================================================================
def check_rate_limit(
    identifier: str,
    bucket: str = "default",
    limit: Optional[int] = None,
    window: Optional[int] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if request is allowed under the rate limit.

    Args:
        identifier: unique key (e.g. IP address, user_id, api_key_hash)
        bucket: named tier from LIMITS
        limit: override max requests
        window: override window in seconds

    Returns:
        (allowed, info_dict)
        info_dict contains: limit, remaining, reset_at (unix ts), retry_after (seconds)
    """
    cfg_limit, cfg_window = LIMITS.get(bucket, LIMITS["default"])
    max_requests = limit if limit is not None else cfg_limit
    window_seconds = window if window is not None else cfg_window

    now = time.time()
    window_start = now - window_seconds
    key = f"ratelimit:{bucket}:{identifier}"

    if _redis_client is not None:
        try:
            # Sliding window with sorted set: timestamp as score AND value
            pipe = _redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)  # purge old
            pipe.zcard(key)                                # count in window
            pipe.zadd(key, {str(now): now})                # add this request
            pipe.expire(key, window_seconds + 5)           # TTL
            _, current_count, _, _ = pipe.execute()

            if current_count >= max_requests:
                # Over limit — remove the just-added entry
                _redis_client.zrem(key, str(now))
                oldest = _redis_client.zrange(key, 0, 0, withscores=True)
                reset_at = oldest[0][1] + window_seconds if oldest else now + window_seconds
                retry_after = max(1, int(reset_at - now))
                return False, {
                    "limit": max_requests,
                    "remaining": 0,
                    "reset_at": int(reset_at),
                    "retry_after": retry_after,
                    "window": window_seconds,
                }

            remaining = max_requests - current_count - 1
            reset_at = int(now + window_seconds)
            return True, {
                "limit": max_requests,
                "remaining": max(0, remaining),
                "reset_at": reset_at,
                "retry_after": 0,
                "window": window_seconds,
            }
        except Exception as e:
            logger.warning(f"Redis rate limit error ({bucket}): {e} — falling back to memory")

    # In-memory fallback
    entries = _memory_store[key] = [t for t in _memory_store[key] if t > window_start]
    if len(entries) >= max_requests:
        oldest = min(entries) if entries else now
        reset_at = int(oldest + window_seconds)
        retry_after = max(1, int(reset_at - now))
        return False, {
            "limit": max_requests,
            "remaining": 0,
            "reset_at": reset_at,
            "retry_after": retry_after,
            "window": window_seconds,
        }
    entries.append(now)
    remaining = max_requests - len(entries)
    return True, {
        "limit": max_requests,
        "remaining": max(0, remaining),
        "reset_at": int(now + window_seconds),
        "retry_after": 0,
        "window": window_seconds,
    }


def build_rate_limit_headers(info: Dict[str, Any]) -> Dict[str, str]:
    """Return the standard rate-limit HTTP headers."""
    headers = {
        "X-RateLimit-Limit": str(info.get("limit", 0)),
        "X-RateLimit-Remaining": str(info.get("remaining", 0)),
        "X-RateLimit-Reset": str(info.get("reset_at", 0)),
    }
    if info.get("retry_after", 0) > 0:
        headers["Retry-After"] = str(info["retry_after"])
    return headers
