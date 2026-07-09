# DeepGuard Pro SaaS — Environment Variables

All variables needed for the SaaS transformation. Add these to `.env` at the
project root. Values marked **required** must be set before deploying the
corresponding module.

---

## Modules 1–3: API Keys, Rate Limiting, Usage Tracking  *(implemented)*

```env
# Redis database used only by the rate limiter (keep separate from Celery DB 0)
REDIS_DB_RATELIMIT=2

# Per-minute rate limits (applied per client IP on /chat_analyze)
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_REQUESTS_PER_MINUTE_UNAUTH=10

# Salt used if you ever move API-key hashing to HMAC-based
# (current implementation uses plain SHA256 — no salt needed yet)
API_KEY_MASTER_SALT=<32-random-chars>
```

---

## Module 7: Security Headers  *(implemented)*

```env
# Set to true ONLY when serving over HTTPS behind a TLS terminator.
# Do not enable on localhost / HTTP — browsers will refuse to connect after caching.
ENABLE_HSTS=false

# Override Content-Security-Policy if you serve embedded widgets or custom domains.
# Default permits self + HTTPS images/media; tighten in production.
CSP_POLICY=<full CSP string>

# CORS — extend beyond the defaults for your production domain + browser extensions
CORS_ORIGINS=http://localhost:5173,http://localhost:8501,https://deepguard.io,https://app.deepguard.io
CORS_ORIGIN_REGEX=^(chrome|moz)-extension://.*$
```

---

## Module 4: Stripe Billing  *(pending — next session)*

```env
# Stripe secret key — use sk_test_... in development, sk_live_... in production
STRIPE_SECRET_KEY=sk_test_xxx

# Webhook endpoint signing secret (from Stripe Dashboard > Webhooks > your endpoint)
STRIPE_WEBHOOK_SECRET=whsec_xxx

# Price IDs created in the Stripe Dashboard (Products > create each plan)
STRIPE_PRO_PRICE_ID=price_xxx
STRIPE_BUSINESS_PRICE_ID=price_yyy

# Frontend URL — used for Stripe Checkout redirect URLs
FRONTEND_URL=http://localhost:8501
API_BASE_URL=http://localhost:8000
```

---

## Module 5: Outbound Webhooks  *(pending)*

```env
# HMAC secret prefix used when signing outbound payloads
WEBHOOK_SIGNATURE_PREFIX=sha256=

# Max failures before a webhook endpoint is auto-disabled
WEBHOOK_MAX_FAILURES=10

# Retry schedule (seconds between retries — comma-separated)
WEBHOOK_RETRY_DELAYS=60,300,900
```

---

## Module 9: Forensic Reports  *(pending)*

```env
# Starting value for the sequential report counter (DG-2026-00001, ...)
REPORTS_SEQUENTIAL_COUNTER_START=1

# Public URL pattern used inside PDFs for the "verify" link
REPORT_VERIFY_URL_BASE=https://deepguard.io/verify
```

---

## Existing variables referenced by new code

Nothing in the new modules changes existing values. Confirm these are still set:

```env
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=deepguard_redis_secret
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=deepguard_password
JWT_SECRET_KEY=<64+ byte random string>
JWT_ALGORITHM=HS256
```
