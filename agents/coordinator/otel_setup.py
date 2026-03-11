"""
OpenTelemetry / SigNoz setup pentru DeepGuard Pro.

Exportă traces, metrics și logs la OTEL Collector (care le trimite la SigNoz).
Se importă o singură dată la startup (api.py și celery_worker).
"""

import os
import logging

logger = logging.getLogger(__name__)

def setup_otel(service_name: str | None = None) -> bool:
    """
    Inițializează OpenTelemetry SDK.
    Returnează True dacă OTEL_EXPORTER_OTLP_ENDPOINT e setat și SDK s-a inițializat.
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    svc = service_name or os.getenv("OTEL_SERVICE_NAME", "deepguard")

    if not endpoint:
        logger.info("⚙️  OTEL: OTEL_EXPORTER_OTLP_ENDPOINT not set — telemetry disabled")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        resource = Resource(attributes={
            SERVICE_NAME: svc,
            "deployment.environment": os.getenv("ENVIRONMENT", "production"),
            "service.version": "1.0.0",
        })

        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        logger.info(f"✅ OTEL Traces: {svc} → {endpoint}")
    except Exception as e:
        logger.warning(f"⚠️  OTEL Traces init failed: {e}")

    # ── Metrics ──────────────────────────────────────────────────────────────
    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint, insecure=True),
            export_interval_millis=30_000,
        )
        meter_provider = MeterProvider(resource=Resource(attributes={SERVICE_NAME: svc}),
                                       metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        logger.info(f"✅ OTEL Metrics: {svc} → {endpoint}")
    except Exception as e:
        logger.warning(f"⚠️  OTEL Metrics init failed: {e}")

    # ── Logs ─────────────────────────────────────────────────────────────────
    try:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
        from opentelemetry.instrumentation.logging import LoggingInstrumentor

        log_provider = LoggerProvider(resource=Resource(attributes={SERVICE_NAME: svc}))
        log_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint, insecure=True))
        )
        set_logger_provider(log_provider)
        LoggingInstrumentor().instrument(set_logging_format=True)
        logger.info(f"✅ OTEL Logs: {svc} → {endpoint}")
    except Exception as e:
        logger.warning(f"⚠️  OTEL Logs init failed: {e}")

    return True


def get_tracer(name: str = "deepguard"):
    """Returnează un tracer OTEL (no-op dacă SDK nu e inițializat)."""
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except Exception:
        return None


def get_meter(name: str = "deepguard"):
    """Returnează un meter OTEL (no-op dacă SDK nu e inițializat)."""
    try:
        from opentelemetry import metrics
        return metrics.get_meter(name)
    except Exception:
        return None
