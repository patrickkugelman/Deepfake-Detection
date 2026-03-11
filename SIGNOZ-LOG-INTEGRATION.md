# SigNoz Observability Integration

## Prezentare Generală

DeepGuard Pro folosește **OpenTelemetry** (standard deschis CNCF) cu **SigNoz** ca backend de observabilitate pentru colectarea și vizualizarea:

- 📊 **Traces** — urmărire request-uri end-to-end (FastAPI → Celery → Agenți)
- 📈 **Metrics** — analize/minut, rată fake/real, latență agenți
- 📋 **Logs** — logs structurate cu trace_id corelat

## Arhitectură

```
DeepGuard Services
  ├── deepguard-coordinator  (FastAPI)
  │       │  OTLP gRPC :4317
  ├── deepguard-celery-worker
  │       │  OTLP gRPC :4317
  └── agenți (vision, audio, video, bio, text, fingerprint)
          │  OTLP gRPC :4317
          ▼
  ┌─────────────────────────────┐
  │   OTEL Collector            │  :4317 (gRPC)
  │   deepguard_otel_collector  │  :4318 (HTTP)
  │                             │  :8889 (Prometheus scrape)
  └─────────┬───────────────────┘
            │
            ▼
  ┌──────────────────┐
  │  SigNoz Cloud    │  sau self-hosted
  │  (ingest EU)     │  http://signoz:3301
  └──────────────────┘
```

## Servicii Docker

| Container | Port | Rol |
|---|---|---|
| `deepguard_otel_collector` | `4317` (gRPC), `4318` (HTTP) | Colectare OTLP |
| `deepguard_otel_collector` | `8889` | Prometheus metrics scrape |
| `deepguard_otel_collector` | `13133` | Health check endpoint |

## Pornire

```bash
docker compose up -d otel-collector
```

Verificare health:

```bash
curl http://localhost:13133
# {"status":"Server available","upSince":"...","uptime":"..."}
```

## Configurare SigNoz Cloud

### 1. Obține Ingestion Key

1. Înregistrează-te la [https://signoz.io](https://signoz.io) (gratuit — 30 zile)
2. Du-te la **Settings → Ingestion Keys**
3. Copiază key-ul

### 2. Setează în `.env`

```env
SIGNOZ_ENDPOINT=https://ingest.eu.signoz.cloud:443
SIGNOZ_INGESTION_KEY=<cheia_ta_de_la_signoz>
```

### 3. Restart collector

```bash
docker compose restart otel-collector
```

## Configurare SigNoz Self-Hosted (opțional)

```bash
# Clonează SigNoz
git clone https://github.com/SigNoz/signoz.git
cd signoz/deploy
docker compose up -d

# Accesează UI la http://localhost:3301
# admin / admin
```

Modifică `.env`:

```env
SIGNOZ_ENDPOINT=http://localhost:4317
SIGNOZ_INGESTION_KEY=
```

## Metrics Custom DeepGuard

| Metric | Tip | Descriere |
|---|---|---|
| `deepguard.analyses.total` | Counter | Total analize submittte |
| `deepguard.verdicts.fake` | Counter | Verdicte FAKE emise |
| `deepguard.verdicts.real` | Counter | Verdicte REAL emise |

Vizualizare în SigNoz: **Dashboards → Create Dashboard → Add Panel → Metric**

Query exemplu:

```
rate(deepguard_analyses_total[5m])
```

## Traces Automate (FastAPIInstrumentor)

Toate request-urile HTTP sunt traceable automat:

| Span | Operație |
|---|---|
| `POST /analyze` | Submit analiză media |
| `GET /job/{id}/status` | Polling status job |
| `POST /chat/{id}` | Conversație Cortex AI |
| `GET /admin/users` | Admin panel |

### Vizualizare Trace în SigNoz

1. Du-te la **Traces → Trace Explorer**
2. Filtrează după `service.name = deepguard-coordinator`
3. Click pe un trace pentru waterfall view

## Logs cu Trace Correlation

Logurile Python sunt automat corelate cu trace-ul activ (prin `LoggingInstrumentor`). Fiecare linie de log conține `trace_id` și `span_id`:

```
2026-03-11 10:23:45 - DeepGuard.Coordinator - INFO - [trace_id=abc123 span_id=def456] 🚀 Celery task dispatched
```

În SigNoz: **Logs → Log Explorer → Filter by trace_id**

## Verificare Funcționare (fără SigNoz)

Colectorul trimite toate datele și în **debug mode** (console). Verifică:

```bash
docker logs deepguard_otel_collector --tail 20
```

Vei vedea spans, metrics și logs afișate în consolă chiar dacă SigNoz nu e configurat.

## Fișiere Configurare

| Fișier | Rol |
|---|---|
| `otel-collector-config.yml` | Configurare OTEL Collector (receivers, processors, exporters) |
| `agents/coordinator/otel_setup.py` | SDK Python inițializare (traces + metrics + logs) |
| `agents/coordinator/requirements.txt` | Pachete OpenTelemetry |

## Referințe

- [OpenTelemetry Python](https://opentelemetry-python.readthedocs.io/)
- [SigNoz Documentation](https://signoz.io/docs/)
- [OTEL Collector Config](https://opentelemetry.io/docs/collector/configuration/)
