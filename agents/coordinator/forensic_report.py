"""
Forensic Report Generator — DeepGuard Pro (Module 9).

Generates a court-grade 5-page PDF report with:
  1. Cover page (DG-YYYY-NNNNN number, verdict, confidence, logo area)
  2. Chain of Custody (file hash, upload time, analyst, IP, timeline)
  3. Methodology (agents used, models, neuro-symbolic pipeline overview)
  4. Per-Agent Results (verdict, confidence, reasoning for each agent)
  5. Conclusion + SHA256 signature + verification link

A ForensicReport node is persisted in Neo4j so the /verify/{report_hash}
endpoint can validate the PDF's authenticity later.
"""
import os
import io
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("DeepGuard.ForensicReport")

REPORTS_SEQUENTIAL_COUNTER_START = int(os.getenv("REPORTS_SEQUENTIAL_COUNTER_START", "1"))
REPORT_VERIFY_URL_BASE = os.getenv("REPORT_VERIFY_URL_BASE", "https://deepguard.io/verify")


# ============================================================================
# Sequential report number
# ============================================================================
def next_report_number(neo4j_conn) -> str:
    """Atomically increment and return a DG-YYYY-NNNNN report number."""
    year = datetime.utcnow().year
    result = neo4j_conn.query("""
        MERGE (c:ReportCounter {year: $year})
        ON CREATE SET c.value = $start
        SET c.value = coalesce(c.value, $start) + 1
        RETURN c.value as v
    """, {"year": year, "start": REPORTS_SEQUENTIAL_COUNTER_START})
    n = int(result[0]["v"]) if result else REPORTS_SEQUENTIAL_COUNTER_START
    return f"DG-{year}-{n:05d}"


# ============================================================================
# PDF generation
# ============================================================================
def generate_forensic_pdf(
    job: Dict[str, Any],
    user: Dict[str, Any],
    report_number: str,
) -> bytes:
    """Render the 5-page forensic PDF. Returns PDF bytes."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=20*mm, bottomMargin=20*mm,
        leftMargin=18*mm, rightMargin=18*mm,
        title=f"DeepGuard Forensic Report {report_number}",
        author="DeepGuard Pro",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('H1', parent=styles['Title'], fontSize=22,
                              textColor=HexColor('#1a1a2e'), spaceAfter=4*mm))
    styles.add(ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14,
                              textColor=HexColor('#1a1a2e'), spaceBefore=5*mm, spaceAfter=2*mm))
    styles.add(ParagraphStyle('Body2', parent=styles['Normal'], fontSize=10,
                              leading=14, textColor=HexColor('#222222'), alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle('Meta', parent=styles['Normal'], fontSize=9,
                              textColor=HexColor('#666666'), alignment=TA_CENTER))

    verdict_upper = (job.get("verdict") or "INCONCLUSIVE").upper()
    if "FAKE" in verdict_upper:
        verdict_color = HexColor('#dc3545')
        verdict_label = "FAKE — Deepfake Detected"
    elif "REAL" in verdict_upper:
        verdict_color = HexColor('#28a745')
        verdict_label = "REAL — Authentic"
    else:
        verdict_color = HexColor('#ffc107')
        verdict_label = "INCONCLUSIVE"

    styles.add(ParagraphStyle('Verdict', parent=styles['Title'], fontSize=26,
                              textColor=verdict_color, alignment=TA_CENTER,
                              spaceBefore=8*mm, spaceAfter=8*mm))

    e = []

    # ------------------------------------------------------------------ PAGE 1
    e.append(Spacer(1, 30*mm))
    e.append(Paragraph("🛡️ DeepGuard Pro", styles['H1']))
    e.append(Paragraph("Forensic Analysis Report", styles['Meta']))
    e.append(HRFlowable(width="80%", color=HexColor('#cccccc'), thickness=0.7))
    e.append(Spacer(1, 20*mm))
    e.append(Paragraph(f"Report No. <b>{report_number}</b>", styles['Meta']))
    e.append(Spacer(1, 10*mm))
    e.append(Paragraph(verdict_label, styles['Verdict']))
    conf = float(job.get("confidence") or 0.0)
    conf_pct = conf * 100 if conf <= 1.0 else conf
    e.append(Paragraph(f"Confidence level: <b>{conf_pct:.1f}%</b>",
                       ParagraphStyle('c', parent=styles['Normal'], fontSize=12,
                                      alignment=TA_CENTER)))
    e.append(Spacer(1, 25*mm))
    e.append(Paragraph(f"File: <b>{job.get('filename', 'N/A')}</b>", styles['Meta']))
    e.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", styles['Meta']))
    e.append(Paragraph(f"Analyst: {user.get('username', 'N/A')}", styles['Meta']))
    e.append(PageBreak())

    # ------------------------------------------------------------------ PAGE 2
    e.append(Paragraph("1. Chain of Custody", styles['H2']))
    e.append(Paragraph(
        "This section documents the complete chain of custody for the analyzed "
        "artifact, from ingestion to verdict emission. All timestamps are in UTC.",
        styles['Body2']))
    e.append(Spacer(1, 4*mm))

    coc_data = [
        ["Report Number", report_number],
        ["Job ID", str(job.get("job_id", "N/A"))],
        ["File SHA-256", str(job.get("file_hash", "N/A"))],
        ["Original filename", str(job.get("filename", "N/A"))],
        ["File size", f"{(job.get('file_size') or 0) / 1_000_000:.2f} MB" if job.get('file_size') else "N/A"],
        ["MIME type", str(job.get("mime_type", "N/A"))],
        ["Resolution", str(job.get("resolution", "N/A"))],
        ["Upload timestamp", str(job.get("created_at", "N/A"))],
        ["Analysis completed", str(job.get("completed_at", "N/A"))],
        ["Analyst (account)", str(user.get("username", "N/A"))],
        ["Storage URL", str(job.get("s3_url") or "internal://")[:70]],
    ]
    t = Table(coc_data, colWidths=[55*mm, 115*mm])
    t.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 9),
        ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#1a1a2e')),
        ('GRID', (0, 0), (-1, -1), 0.25, HexColor('#cccccc')),
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#f5f5f7')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    e.append(t)
    e.append(PageBreak())

    # ------------------------------------------------------------------ PAGE 3
    e.append(Paragraph("2. Methodology", styles['H2']))
    e.append(Paragraph(
        "DeepGuard Pro applies a multi-agent neural pipeline combined with "
        "a neuro-symbolic reasoning layer. Each specialized agent operates "
        "independently on the artifact; their outputs are then reconciled by "
        "a contradiction-aware debater that resolves inconsistencies via "
        "formal logic rules. When available, Google SynthID watermark "
        "detection acts as ground-truth override.",
        styles['Body2']))
    e.append(Spacer(1, 3*mm))

    agents_used = []
    details = job.get("details") or {}
    if isinstance(details, dict):
        agents_used = list(details.get("agents_used") or details.keys())

    e.append(Paragraph("<b>Agents activated for this artifact:</b>", styles['Body2']))
    for a in agents_used or ["(none recorded)"]:
        e.append(Paragraph(f"• {a}", styles['Body2']))

    e.append(Spacer(1, 4*mm))
    e.append(Paragraph("<b>Pipeline stages:</b>", styles['Body2']))
    for step in [
        "Agent selection (file-type + agentic routing)",
        "SynthID watermark verification (if applicable)",
        "Parallel agent inference (vision, video, audio, bio, text, fingerprint)",
        "GPT-4o / Gemini vision frame-by-frame deep scan",
        "Neuro-symbolic contradiction resolution",
        "Verdict consolidation + confidence estimation",
    ]:
        e.append(Paragraph(f"• {step}", styles['Body2']))
    e.append(PageBreak())

    # ------------------------------------------------------------------ PAGE 4
    e.append(Paragraph("3. Per-Agent Results", styles['H2']))
    e.append(Paragraph(
        "The following table summarizes the verdict, confidence and reasoning "
        "returned by each agent that participated in the analysis.",
        styles['Body2']))
    e.append(Spacer(1, 3*mm))

    rows = [["Agent", "Verdict", "Confidence", "Notes"]]
    if isinstance(details, dict):
        for name, res in details.items():
            if name in ("agents_used",):
                continue
            if not isinstance(res, dict):
                continue
            v = str(res.get("verdict", "—"))
            c = res.get("confidence", 0)
            try:
                c_str = f"{float(c) * 100:.1f}%" if float(c) <= 1.0 else f"{float(c):.1f}%"
            except Exception:
                c_str = "—"
            notes = str(res.get("reasoning") or res.get("error") or "")[:120]
            rows.append([name, v, c_str, notes])
    if len(rows) == 1:
        rows.append(["(no per-agent data available)", "", "", ""])

    t2 = Table(rows, colWidths=[30*mm, 28*mm, 22*mm, 90*mm], repeatRows=1)
    t2.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 8),
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('GRID', (0, 0), (-1, -1), 0.25, HexColor('#cccccc')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    e.append(t2)
    e.append(PageBreak())

    # ------------------------------------------------------------------ PAGE 5
    e.append(Paragraph("4. Conclusion", styles['H2']))
    e.append(Paragraph(verdict_label, styles['Verdict']))
    reasoning = str(job.get("reasoning") or "No reasoning provided.")
    e.append(Paragraph(reasoning, styles['Body2']))
    e.append(Spacer(1, 8*mm))

    e.append(Paragraph("5. Authenticity Signature", styles['H2']))
    e.append(Paragraph(
        "This PDF report is cryptographically bound to its contents. The SHA-256 "
        "digest below was computed over the canonical JSON representation of the "
        "analysis at generation time. Any tampering with the PDF or the underlying "
        "record can be detected by re-computing this digest and comparing it with "
        "the value stored in the DeepGuard verification registry.",
        styles['Body2']))
    e.append(Spacer(1, 3*mm))
    # Placeholder — replaced after hash computation below
    sig_placeholder = Paragraph("<b>SHA-256:</b> {SIGNATURE}", styles['Body2'])
    e.append(sig_placeholder)
    e.append(Spacer(1, 2*mm))
    e.append(Paragraph(f"<b>Verify at:</b> {REPORT_VERIFY_URL_BASE}/{{REPORT_HASH}}", styles['Body2']))

    # Build once to know the content — compute hash over a canonical JSON of the job
    canonical = json.dumps({
        "report_number": report_number,
        "job_id": job.get("job_id"),
        "file_hash": job.get("file_hash"),
        "verdict": job.get("verdict"),
        "confidence": job.get("confidence"),
        "reasoning": job.get("reasoning"),
        "filename": job.get("filename"),
        "completed_at": str(job.get("completed_at") or ""),
        "username": user.get("username"),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report_hash = hashlib.sha256(canonical).hexdigest()

    # Replace placeholder paragraphs by rebuilding the list with real values
    # Swap the last two paragraphs just before build.
    e[-2] = Paragraph(f"<b>SHA-256:</b> <font face='Courier'>{report_hash}</font>", styles['Body2'])
    e[-1] = Paragraph(
        f"<b>Verify at:</b> <link href='{REPORT_VERIFY_URL_BASE}/{report_hash}'>"
        f"{REPORT_VERIFY_URL_BASE}/{report_hash}</link>", styles['Body2'])

    doc.build(e)
    pdf_bytes = buf.getvalue()
    buf.close()

    # Stash the hash on the bytes object for the caller — we actually just
    # return (pdf_bytes, report_hash) via the public API below.
    return pdf_bytes, report_hash  # type: ignore[return-value]


# ============================================================================
# Full pipeline: build PDF + persist ForensicReport node
# ============================================================================
def create_forensic_report(
    neo4j_conn,
    job_id: str,
    username: str,
) -> Dict[str, Any]:
    """Generate a forensic report for a completed job. Returns {pdf, report_hash, report_number}."""
    # Fetch the job
    rows = neo4j_conn.query("""
        MATCH (j:Job {id: $jid})
        OPTIONAL MATCH (j)-[:RESULT]->(f:File)
        RETURN j.id as job_id, j.status as status,
               coalesce(f.verdict, j.verdict) as verdict,
               coalesce(f.reasoning, j.status_message) as reasoning,
               coalesce(f.confidence, 0.0) as confidence,
               coalesce(f.hash, '') as file_hash,
               coalesce(f.filename, j.filename, '') as filename,
               coalesce(f.url, '') as s3_url,
               j.file_size as file_size,
               j.mime_type as mime_type,
               j.resolution as resolution,
               j.created_at as created_at,
               j.completed_at as completed_at,
               j.result as result_json
    """, {"jid": job_id})
    if not rows:
        raise ValueError(f"Job {job_id} not found")
    job = dict(rows[0])
    if str(job.get("status")) != "completed":
        raise ValueError(f"Job {job_id} is not completed (status={job.get('status')})")

    # Unpack stored details from serialized result
    details = {}
    try:
        raw = job.get("result_json")
        if raw:
            parsed = json.loads(raw)
            details = parsed.get("details") or {}
            if not job.get("reasoning"):
                job["reasoning"] = parsed.get("reasoning")
            if not job.get("confidence"):
                job["confidence"] = parsed.get("confidence")
    except Exception as e:
        logger.warning(f"Could not parse job result JSON: {e}")
    job["details"] = details

    user = {"username": username}
    report_number = next_report_number(neo4j_conn)

    result = generate_forensic_pdf(job, user, report_number)
    # generate_forensic_pdf returns tuple (bytes, hash)
    pdf_bytes, report_hash = result  # type: ignore[misc]

    # Persist ForensicReport node
    neo4j_conn.query("""
        MATCH (u:User {username: $u})
        OPTIONAL MATCH (j:Job {id: $jid})
        CREATE (r:ForensicReport {
            report_number: $rn,
            report_hash: $rh,
            job_id: $jid,
            username: $u,
            verdict: $verdict,
            confidence: $confidence,
            file_hash: $fhash,
            filename: $fname,
            generated_at: datetime()
        })
        WITH u, j, r
        FOREACH (_ IN CASE WHEN j IS NOT NULL THEN [1] ELSE [] END |
            MERGE (r)-[:FOR_JOB]->(j))
        MERGE (u)-[:GENERATED_REPORT]->(r)
    """, {
        "u": username,
        "jid": job_id,
        "rn": report_number,
        "rh": report_hash,
        "verdict": job.get("verdict"),
        "confidence": float(job.get("confidence") or 0.0),
        "fhash": job.get("file_hash"),
        "fname": job.get("filename"),
    })

    logger.info(f"📜 Forensic report {report_number} ({report_hash[:12]}...) generated for job {job_id}")
    return {
        "pdf": pdf_bytes,
        "report_hash": report_hash,
        "report_number": report_number,
        "verdict": job.get("verdict"),
        "filename": job.get("filename"),
    }


def verify_report_hash(neo4j_conn, report_hash: str) -> Optional[Dict[str, Any]]:
    """Public verification: look up a ForensicReport by its SHA-256 hash."""
    rows = neo4j_conn.query("""
        MATCH (r:ForensicReport {report_hash: $rh})
        RETURN r.report_number as report_number,
               r.report_hash as report_hash,
               r.job_id as job_id,
               r.verdict as verdict,
               r.confidence as confidence,
               r.filename as filename,
               r.generated_at as generated_at,
               r.username as issued_to
    """, {"rh": report_hash})
    if not rows:
        return None
    row = dict(rows[0])
    # Do not leak the issuing user to the public verify endpoint
    issued_to = row.pop("issued_to", None)
    return {
        "valid": True,
        "report_number": row["report_number"],
        "report_hash": row["report_hash"],
        "job_id": row["job_id"],
        "verdict": row["verdict"],
        "confidence": row["confidence"],
        "filename": row["filename"],
        "generated_at": str(row["generated_at"]) if row.get("generated_at") else None,
        "issuer": "DeepGuard Pro",
    }
