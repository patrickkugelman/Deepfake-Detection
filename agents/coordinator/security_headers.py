"""
Security headers middleware for DeepGuard Pro.

Adds standard security headers to every response:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Referrer-Policy: strict-origin-when-cross-origin
    - Strict-Transport-Security (when served via HTTPS)
    - Content-Security-Policy (permissive default, override via env)
"""
import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Override via env if the defaults are too restrictive for a particular deploy.
CSP_POLICY = os.getenv(
    "CSP_POLICY",
    "default-src 'self'; "
    "img-src 'self' data: blob: https:; "
    "media-src 'self' blob: https:; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
    "style-src 'self' 'unsafe-inline'; "
    "connect-src 'self' https: wss:; "
    "frame-ancestors 'none';"
)
ENABLE_HSTS = os.getenv("ENABLE_HSTS", "false").lower() == "true"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject standard security headers on every response."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        # Always set these
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-XSS-Protection", "1; mode=block")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), camera=(), microphone=()")

        # CSP (skip for /docs and /redoc — they need inline scripts from the FastAPI UI)
        path = request.url.path
        if not path.startswith("/docs") and not path.startswith("/redoc") and not path.startswith("/openapi"):
            response.headers.setdefault("Content-Security-Policy", CSP_POLICY)

        # HSTS only over HTTPS (opt-in via env to avoid breaking local dev)
        if ENABLE_HSTS and request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains"
            )

        return response
