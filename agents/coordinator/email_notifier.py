import smtplib
import os
import logging
import urllib.request
import urllib.error
import json as _json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime
from typing import Optional

logger = logging.getLogger("DeepGuard.Email")

class EmailNotifier:
    def __init__(self):
        self.server = os.getenv("MAIL_SERVER", "smtp-relay.brevo.com")
        self.port = int(os.getenv("MAIL_PORT", "587"))
        self.user = os.getenv("MAIL_USERNAME", "")
        self.password = os.getenv("MAIL_PASSWORD", "")
        self.admin_email = os.getenv("ADMIN_EMAIL", "")
        self.use_tls = os.getenv("MAIL_USE_TLS", "true").lower() == "true"
        self.from_email = os.getenv("MAIL_FROM", self.user)
        # n8n webhook — dacă e setat, emailurile se trimit prin n8n (Gmail) în loc de SMTP direct
        self.n8n_webhook = os.getenv("N8N_WEBHOOK_URL", "").strip()
        self.n8n_osint_webhook = os.getenv("N8N_OSINT_WEBHOOK_URL", "").strip()
        if self.n8n_webhook:
            logger.info(f"📨 Email mode: n8n webhook ({self.n8n_webhook})")
        else:
            logger.info(f"📨 Email mode: SMTP direct ({self.server}:{self.port})")
        if self.n8n_osint_webhook:
            logger.info(f"🔍 OSINT webhook configured: {self.n8n_osint_webhook}")

    def _send_via_n8n(self, payload: dict) -> bool:
        """Trimite payload JSON la n8n webhook. Returnează True dacă reușește."""
        try:
            data = _json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.n8n_webhook,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info(f"📨 n8n webhook OK ({resp.status}) → {payload.get('to')}")
                return True
        except Exception as e:
            logger.error(f"❌ n8n webhook failed: {e}")
            return False

    def send_login_alert(self, recipient_email: str, username: str):
        """Trimite o notificare de securitate la logare."""
        # --- n8n webhook ---
        if self.n8n_webhook:
            return self._send_via_n8n({
                "type": "login_alert",
                "to": recipient_email,
                "username": username,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
        # --- fallback SMTP ---
        try:
            msg = MIMEMultipart()
            msg['From'] = f"DeepGuard Security <{self.from_email}>"
            msg['To'] = recipient_email
            msg['Subject'] = "🔐 Alertă de Securitate: Logare Nouă"
            msg['MIME-Version'] = "1.0"
            msg['Content-Type'] = "multipart/alternative"

            body = f"""
            <!DOCTYPE html>
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                    <h2 style="color: #0056b3;">Logare Detectată</h2>
                    <p>Salut <strong>{username}</strong>,</p>
                    <p>O nouă logare a fost detectată pe contul tău DeepGuard Pro.</p>
                    <p><strong>Utilizator:</strong> {username}</p>
                    <p><strong>Data:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <hr>
                    <p style="font-size: 12px; color: #999;">Dacă nu ai fost tu, contactează administratorul.</p>
                </div>
            </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))

            with smtplib.SMTP(self.server, self.port) as server:
                if self.use_tls or self.port == 587:
                    server.starttls()
                if self.password:
                    server.login(self.user, self.password)
                server.send_message(msg)

            logger.info(f"📧 Login alert sent to {recipient_email}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send login alert: {e}")
            return False

    def _generate_pdf_report(self, filename: str, verdict: str, reasoning: str, confidence: float = 0.0) -> Optional[bytes]:
        """Generate a PDF report and return bytes. Returns None if reportlab unavailable."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import mm
            from reportlab.lib.colors import HexColor
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER
            import io

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    topMargin=25*mm, bottomMargin=20*mm,
                                    leftMargin=20*mm, rightMargin=20*mm)
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name='Title2', parent=styles['Title'],
                                      fontSize=22, spaceAfter=4*mm, textColor=HexColor('#1a1a2e')))
            styles.add(ParagraphStyle(name='Subtitle', parent=styles['Normal'],
                                      fontSize=11, textColor=HexColor('#666666'),
                                      spaceAfter=8*mm, alignment=TA_CENTER))
            styles.add(ParagraphStyle(name='Section', parent=styles['Heading2'],
                                      fontSize=14, textColor=HexColor('#1a1a2e'),
                                      spaceBefore=6*mm, spaceAfter=3*mm))
            styles.add(ParagraphStyle(name='Body2', parent=styles['Normal'],
                                      fontSize=10, leading=14, textColor=HexColor('#333333')))

            v = verdict.upper()
            if "FAKE" in v:
                vc = HexColor('#dc3545'); vl = "FAKE — Deepfake Detectat"
            elif "REAL" in v:
                vc = HexColor('#28a745'); vl = "REAL — Autentic"
            else:
                vc = HexColor('#ffc107'); vl = "INCONCLUSIVE — Neconcludent"

            styles.add(ParagraphStyle(name='Verdict', parent=styles['Title'],
                                      fontSize=20, textColor=vc, alignment=TA_CENTER,
                                      spaceBefore=4*mm, spaceAfter=4*mm))

            elems = []
            elems.append(Paragraph("🛡️ DeepGuard Pro", styles['Title2']))
            elems.append(Paragraph("Raport de Analiză Forensic — Neural Multi-Agent Detection System", styles['Subtitle']))
            elems.append(HRFlowable(width="100%", color=HexColor('#cccccc'), thickness=0.5))
            elems.append(Spacer(1, 4*mm))
            elems.append(Paragraph("Verdict Final", styles['Section']))
            elems.append(Paragraph(vl, styles['Verdict']))
            elems.append(Paragraph(f"Nivel de Încredere: <b>{confidence:.1f}%</b>", styles['Body2']))
            elems.append(Spacer(1, 4*mm))

            table_data = [
                ["Fișier:", filename],
                ["Data Raport:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ]
            t = Table(table_data, colWidths=[35*mm, 130*mm])
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#555555')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3*mm),
            ]))
            elems.append(t)
            elems.append(Spacer(1, 4*mm))

            elems.append(Paragraph("Analiză Neuro-Simbolică", styles['Section']))
            reasoning_safe = (reasoning or "N/A").replace('\n', '<br/>')
            elems.append(Paragraph(reasoning_safe, styles['Body2']))
            elems.append(Spacer(1, 6*mm))

            elems.append(HRFlowable(width="100%", color=HexColor('#cccccc'), thickness=0.5))
            footer = ParagraphStyle(name='Footer', parent=styles['Normal'],
                                     fontSize=8, textColor=HexColor('#999999'), alignment=TA_CENTER)
            elems.append(Spacer(1, 2*mm))
            elems.append(Paragraph(
                "Raport generat automat de DeepGuard Pro AI. "
                "Rezultatele au caracter informativ și nu constituie o expertiză legală.", footer))

            doc.build(elems)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            logger.warning(f"⚠️ Could not generate PDF report: {e}")
            return None

    def send_analysis_complete(self, recipient_email: str, filename: str, verdict: str, reasoning: str, job_id: str = "", confidence: float = 0.0):
        """Trimite un mail utilizatorului cu rezultatul analizei + link direct + PDF atașat."""
        # --- n8n webhook ---
        if self.n8n_webhook:
            base_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
            results_url = f"{base_url}/results/job-{job_id}" if job_id else f"{base_url}/dashboard"
            return self._send_via_n8n({
                "type": "analysis_complete",
                "to": recipient_email,
                "filename": filename,
                "verdict": verdict,
                "confidence": round(confidence, 1),
                "reasoning": reasoning,
                "job_id": job_id,
                "results_url": results_url,
            })
        # --- fallback SMTP ---
        try:
            msg = MIMEMultipart("mixed")
            msg['From'] = f"DeepGuard Pro <{self.from_email}>"
            msg['To'] = recipient_email
            msg['Subject'] = f"✅ Analiză Finalizată: {filename}"
            msg['MIME-Version'] = "1.0"

            # Determine colors based on verdict
            if "FAKE" in verdict.upper():
                verdict_color = "#dc3545"
                border_color = "#dc3545"
                verdict_icon = "🔴"
            elif "REAL" in verdict.upper():
                verdict_color = "#28a745"
                border_color = "#28a745"
                verdict_icon = "🟢"
            else:
                verdict_color = "#ffc107"
                border_color = "#ffc107"
                verdict_icon = "🟡"

            # Build results URL - direct link to job results
            base_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
            if job_id:
                results_url = f"{base_url}/results/job-{job_id}"
            else:
                results_url = f"{base_url}/dashboard"

            body = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta charset="utf-8">
            <title>Analiză Finalizată</title>
            </head>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f5f5f5; margin: 0; padding: 20px;">
                <div style="max-width: 600px; margin: 0 auto; padding: 30px; background: white; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">

                    <div style="text-align: center; margin-bottom: 25px;">
                        <h2 style="color: #2c3e50; margin-bottom: 5px;">🛡️ DeepGuard Pro</h2>
                        <p style="color: #888; margin-top: 0;">Analiza ta a fost finalizată cu succes</p>
                    </div>

                    <p>Salut,</p>
                    <p>Sistemul nostru a finalizat expertiza pentru fișierul:</p>
                    <p style="background: #f0f0f0; padding: 8px 14px; border-radius: 6px; font-family: monospace;"><strong>{filename}</strong></p>

                    <div style="padding: 20px; background-color: #f8f9fa; border-left: 5px solid {border_color}; margin: 20px 0; border-radius: 0 8px 8px 0;">
                        <h3 style="margin-top: 0; color: {verdict_color};">{verdict_icon} Verdict: {verdict}</h3>
                        <p style="margin-bottom: 0;"><strong>Motivare:</strong> {reasoning}</p>
                    </div>

                    <p style="color: #555;">Pentru a vedea raportul complet cu toate detaliile (heatmap, scoruri per agent, evidențe), conectează-te în aplicație:</p>

                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{results_url}" style="background-color: #007bff; color: white; padding: 14px 32px; text-decoration: none; border-radius: 6px; font-weight: bold; font-size: 16px; display: inline-block;">🔍 Vezi Raportul Complet</a>
                    </div>

                    <p style="color: #888; font-size: 13px; text-align: center;">Dacă sesiunea ta a expirat, vei fi redirecționat către pagina de login.<br>După autentificare, vei putea accesa rezultatele analizei.</p>

                    <hr style="margin-top: 30px; border: 0; border-top: 1px solid #eee;">
                    <p style="text-align: center; font-size: 11px; color: #aaa;">Acesta este un mesaj automat generat de DeepGuard Pro AI.<br>Analiza a rulat în background și nu necesită tab-ul deschis.</p>
                </div>
            </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))

            # Attach PDF report
            pdf_bytes = self._generate_pdf_report(filename, verdict, reasoning, confidence)
            if pdf_bytes:
                pdf_attachment = MIMEApplication(pdf_bytes, _subtype='pdf')
                safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
                pdf_attachment.add_header('Content-Disposition', 'attachment',
                                          filename=f'DeepGuard_Report_{safe_name}.pdf')
                msg.attach(pdf_attachment)
                logger.info(f"📎 PDF report attached ({len(pdf_bytes)} bytes)")

            with smtplib.SMTP(self.server, self.port) as server:
                if self.use_tls or self.port == 587:
                    server.starttls()
                if self.password:
                    server.login(self.user, self.password)
                server.send_message(msg)
            logger.info(f"📧 Email trimis cu succes către {recipient_email} (job: {job_id})")
            return True
        except Exception as e:
            logger.error(f"❌ Eșec la trimiterea email-ului: {e}")
            return False

    def send_welcome_email(self, recipient_email: str, username: str, password_hint: Optional[str] = None):
        """Trimite un email de bun venit la înregistrare. password_hint: optional, pentru demo/testing."""
        # --- n8n webhook ---
        if self.n8n_webhook:
            base_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
            return self._send_via_n8n({
                "type": "welcome",
                "to": recipient_email,
                "username": username,
                "password_hint": password_hint or "",
                "login_url": f"{base_url}/login",
            })
        # --- fallback SMTP ---
        try:
            msg = MIMEMultipart()
            msg['From'] = f"DeepGuard Pro <{self.from_email}>"
            msg['To'] = recipient_email
            msg['Subject'] = "👋 Bun venit în DeepGuard Pro!"
            msg['MIME-Version'] = "1.0"
            msg['Content-Type'] = "multipart/alternative"

            login_instructions = f"Te poți autentifica cu username-ul <strong>{username}</strong> (sau email) și parola pe care ai folosit-o la înregistrare."
            if password_hint:
                login_instructions = f"Te poți autentifica cu username-ul <strong>{username}</strong> și parola: <code>{password_hint}</code>"

            body = f"""
            <!DOCTYPE html>
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                    <h2 style="color: #0056b3;">Bun venit, {username}!</h2>
                    <p>Contul tău DeepGuard Pro a fost creat cu succes.</p>
                    <p>Acum poți accesa platforma pentru a detecta conținutul deepfake cu ajutorul inteligenței artificiale.</p>
                    <p>{login_instructions}</p>
                    <div style="text-align: center; margin-top: 30px;">
                        <a href="{os.getenv('FRONTEND_URL', 'http://localhost:8501')}/login" style="background-color: #0056b3; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; font-weight: bold;">Accesează Contul</a>
                    </div>
                    <hr style="margin-top: 30px; border: 0; border-top: 1px solid #eee;">
                    <p style="font-size: 12px; color: #999;">Dacă nu ai creat acest cont, ignoră acest mesaj.</p>
                </div>
            </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))

            with smtplib.SMTP(self.server, self.port) as server:
                if self.use_tls or self.port == 587:
                    server.starttls()
                if self.password:
                    server.login(self.user, self.password)
                server.send_message(msg)
            
            logger.info(f"📧 Welcome email sent to {recipient_email}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send welcome email: {e}")
            return False

    def send_osint_trigger(self, job_id: str, filename: str, verdict: str,
                           confidence: float = 0.0, file_hash: str = "",
                           user_email: str = "", s3_url: str = "") -> bool:
        """
        Declanșează un workflow n8n de investigație OSINT automat.
        Se apelează doar când verdict == 'FAKE'.
        Trimite un payload detaliat la N8N_OSINT_WEBHOOK_URL.
        """
        if not self.n8n_osint_webhook:
            logger.debug("🔍 OSINT webhook not configured — skipping OSINT trigger")
            return False
        if verdict != "FAKE":
            return False

        base_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
        results_url = f"{base_url}/results/{job_id}"

        payload = {
            "type": "osint_investigation",
            "job_id": job_id,
            "filename": filename,
            "verdict": verdict,
            "confidence": round(confidence, 1),
            "file_hash": file_hash,
            "user_email": user_email,
            "s3_url": s3_url,
            "results_url": results_url,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "investigation_tasks": [
                "reverse_image_search",
                "check_synthetic_media_db",
                "timeline_analysis",
                "generate_osint_report"
            ]
        }

        try:
            data = _json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.n8n_osint_webhook,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info(f"🔍 OSINT webhook OK ({resp.status}) for job {job_id}")
                return True
        except Exception as e:
            logger.error(f"❌ OSINT webhook failed: {e}")
            return False