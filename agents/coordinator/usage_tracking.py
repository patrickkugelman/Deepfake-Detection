"""
Advanced usage tracking for DeepGuard SaaS.

Builds on top of api_keys.py (which tracks monthly totals on User)
by adding per-day, per-key granularity (DailyUsage node) for dashboards
and charts.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

logger = logging.getLogger("DeepGuard.UsageTracking")


# ============================================================================
# DailyUsage node (per user, per day)
# ============================================================================
def record_daily_usage(
    neo4j_conn,
    username: str,
    api_key_prefix: str = "",
    verdict: str = "UNKNOWN",
    file_type: str = "unknown",
) -> None:
    """
    Upsert a DailyUsage node for today, incrementing relevant counters.

    Called AFTER an analysis successfully completes, alongside
    api_keys.increment_usage().

    Node properties:
        date:              YYYY-MM-DD
        username:          str
        api_key_prefix:    str (empty if JWT auth was used)
        request_count:     int
        real_count:        int
        fake_count:        int
        inconclusive_count:int
        image_count:       int
        video_count:       int
        audio_count:       int
        text_count:        int
    """
    today = datetime.now().strftime("%Y-%m-%d")
    verdict_upper = (verdict or "").upper()

    verdict_field = {
        "REAL": "real_count",
        "FAKE": "fake_count",
        "INCONCLUSIVE": "inconclusive_count",
        "SUSPICIOUS": "inconclusive_count",
    }.get(verdict_upper, "inconclusive_count")

    ftype = (file_type or "").lower()
    if ftype.startswith("image") or ftype in ("jpg", "jpeg", "png", "gif", "webp"):
        type_field = "image_count"
    elif ftype.startswith("video") or ftype in ("mp4", "mov", "avi", "mkv", "webm"):
        type_field = "video_count"
    elif ftype.startswith("audio") or ftype in ("mp3", "wav", "ogg", "flac", "m4a"):
        type_field = "audio_count"
    elif ftype.startswith("text") or ftype in ("txt",):
        type_field = "text_count"
    else:
        type_field = "image_count"  # default bucket

    try:
        neo4j_conn.query(f"""
            MERGE (d:DailyUsage {{date: $date, username: $username, api_key_prefix: $prefix}})
            ON CREATE SET
                d.request_count = 1,
                d.real_count = 0,
                d.fake_count = 0,
                d.inconclusive_count = 0,
                d.image_count = 0,
                d.video_count = 0,
                d.audio_count = 0,
                d.text_count = 0,
                d.created_at = datetime(),
                d.{verdict_field} = 1,
                d.{type_field} = 1
            ON MATCH SET
                d.request_count = d.request_count + 1,
                d.{verdict_field} = coalesce(d.{verdict_field}, 0) + 1,
                d.{type_field} = coalesce(d.{type_field}, 0) + 1
        """, {"date": today, "username": username, "prefix": api_key_prefix or ""})
    except Exception as e:
        logger.error(f"Failed to record daily usage for {username}: {e}")


def get_daily_chart(neo4j_conn, username: str, days: int = 30) -> List[Dict[str, Any]]:
    """
    Return request counts per day for the last N days.
    Missing days are filled with zero.
    """
    try:
        results = neo4j_conn.query("""
            MATCH (d:DailyUsage {username: $username})
            WHERE d.date >= $start_date
            RETURN d.date as date,
                   d.request_count as requests,
                   d.real_count as real,
                   d.fake_count as fake,
                   d.inconclusive_count as inconclusive
            ORDER BY d.date ASC
        """, {
            "username": username,
            "start_date": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
        })
    except Exception as e:
        logger.error(f"get_daily_chart failed: {e}")
        results = []

    by_date = {r["date"]: r for r in (results or [])}
    chart = []
    for i in range(days, -1, -1):
        day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        if day in by_date:
            r = by_date[day]
            chart.append({
                "date": day,
                "requests": int(r.get("requests") or 0),
                "real": int(r.get("real") or 0),
                "fake": int(r.get("fake") or 0),
                "inconclusive": int(r.get("inconclusive") or 0),
            })
        else:
            chart.append({"date": day, "requests": 0, "real": 0, "fake": 0, "inconclusive": 0})
    return chart


def get_current_month_breakdown(neo4j_conn, username: str) -> Dict[str, Any]:
    """
    Aggregate verdict + file-type breakdown for the current calendar month.
    """
    try:
        month_prefix = datetime.now().strftime("%Y-%m")
        results = neo4j_conn.query("""
            MATCH (d:DailyUsage {username: $username})
            WHERE d.date STARTS WITH $month
            RETURN
                sum(d.request_count) as total,
                sum(d.real_count) as real,
                sum(d.fake_count) as fake,
                sum(d.inconclusive_count) as inconclusive,
                sum(d.image_count) as image,
                sum(d.video_count) as video,
                sum(d.audio_count) as audio,
                sum(d.text_count) as text
        """, {"username": username, "month": month_prefix})
        if not results:
            return _empty_breakdown()
        row = results[0]
        return {
            "total_requests": int(row.get("total") or 0),
            "verdicts": {
                "real": int(row.get("real") or 0),
                "fake": int(row.get("fake") or 0),
                "inconclusive": int(row.get("inconclusive") or 0),
            },
            "by_file_type": {
                "image": int(row.get("image") or 0),
                "video": int(row.get("video") or 0),
                "audio": int(row.get("audio") or 0),
                "text": int(row.get("text") or 0),
            },
        }
    except Exception as e:
        logger.error(f"get_current_month_breakdown failed: {e}")
        return _empty_breakdown()


def _empty_breakdown() -> Dict[str, Any]:
    return {
        "total_requests": 0,
        "verdicts": {"real": 0, "fake": 0, "inconclusive": 0},
        "by_file_type": {"image": 0, "video": 0, "audio": 0, "text": 0},
    }


def get_per_key_usage(neo4j_conn, username: str) -> List[Dict[str, Any]]:
    """
    Return per-key usage for the current month (requests per API key).
    """
    try:
        month_prefix = datetime.now().strftime("%Y-%m")
        results = neo4j_conn.query("""
            MATCH (d:DailyUsage {username: $username})
            WHERE d.date STARTS WITH $month AND d.api_key_prefix <> ''
            RETURN d.api_key_prefix as prefix,
                   sum(d.request_count) as requests
            ORDER BY requests DESC
        """, {"username": username, "month": month_prefix})
        return [
            {"api_key_prefix": r["prefix"], "requests_this_month": int(r.get("requests") or 0)}
            for r in (results or [])
        ]
    except Exception as e:
        logger.error(f"get_per_key_usage failed: {e}")
        return []


# ============================================================================
# Monthly reset (called by Celery Beat on day 1 @ 00:00)
# ============================================================================
def reset_all_monthly_quotas(neo4j_conn) -> int:
    """
    Reset monthly_usage to 0 for ALL users. Returns number of users updated.
    Called on day 1 of each month by Celery Beat.
    """
    try:
        current_month = datetime.now().month
        result = neo4j_conn.query("""
            MATCH (u:User)
            SET u.monthly_usage = 0,
                u.usage_reset_month = $month,
                u.last_monthly_reset = datetime()
            RETURN count(u) as updated
        """, {"month": current_month})
        count = int(result[0].get("updated", 0)) if result else 0
        logger.info(f"🔄 Monthly quota reset: {count} users")
        return count
    except Exception as e:
        logger.error(f"Monthly reset failed: {e}")
        return 0


def cleanup_old_daily_usage(neo4j_conn, keep_days: int = 180) -> int:
    """Delete DailyUsage nodes older than keep_days."""
    try:
        cutoff = (datetime.now() - timedelta(days=keep_days)).strftime("%Y-%m-%d")
        result = neo4j_conn.query("""
            MATCH (d:DailyUsage)
            WHERE d.date < $cutoff
            DETACH DELETE d
            RETURN count(d) as deleted
        """, {"cutoff": cutoff})
        count = int(result[0].get("deleted", 0)) if result else 0
        logger.info(f"🧹 Cleaned {count} old DailyUsage nodes")
        return count
    except Exception as e:
        logger.error(f"DailyUsage cleanup failed: {e}")
        return 0
