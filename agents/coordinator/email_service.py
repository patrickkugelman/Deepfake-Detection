"""
Email Service - Welcome & Login Alerts
======================================
Sends welcome emails on registration and login alerts.
Configure via .env: MAIL_SERVER, MAIL_PORT, MAIL_USERNAME, MAIL_PASSWORD
"""

import smtplib
import os
import urllib.request
import json as _json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("MAIL_SERVER", "smtp-relay.brevo.com")
        self.smtp_port = int(os.getenv("MAIL_PORT", "587"))
        self.username = os.getenv("MAIL_USERNAME", "")
        self.password = os.getenv("MAIL_PASSWORD", "")
        self.use_tls = os.getenv("MAIL_USE_TLS", "true").lower() == "true"
        self.from_email = os.getenv("MAIL_FROM", self.username)
        self.n8n_webhook = os.getenv("N8N_WEBHOOK_URL", "").strip()
        self.enabled = bool(self.n8n_webhook or (self.username and self.password))

    def _send_via_n8n(self, payload: dict) -> bool:
        try:
            data = _json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.n8n_webhook,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10):
                return True
        except Exception as e:
            print(f"n8n webhook error: {e}")
            return False
    
    def send_welcome_email(self, to_email: str, username: str):
        if not self.enabled:
            return
        if self.n8n_webhook:
            self._send_via_n8n({"type": "welcome", "to": to_email, "username": username, "password_hint": "", "login_url": os.getenv("FRONTEND_URL", "http://localhost:5173") + "/login"})
            return
        subject = "Bine ai venit la DeepGuard Pro! 🛡️"
        body = f"Salut {username},\n\nContul tău a fost creat!\nEmail: {to_email}"
        self._send(to_email, subject, body)

    def send_login_alert(self, to_email: str, username: str, timestamp: str):
        if not self.enabled:
            return
        if self.n8n_webhook:
            self._send_via_n8n({"type": "login_alert", "to": to_email, "username": username, "timestamp": timestamp})
            return
        subject = "Login Alert - DeepGuard Pro"
        body = f"Salut {username},\n\nAutentificare nouă la {timestamp}"
        self._send(to_email, subject, body)
    
    def _send(self, to: str, subject: str, body: str):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls or self.smtp_port == 587:
                server.starttls()
            if self.password:
                server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"Email error: {e}")


email_service = EmailService()
