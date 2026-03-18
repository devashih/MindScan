import smtplib
from email.message import EmailMessage

# -------------------------------------------------
# SYSTEM EMAIL (APP EMAIL)
# -------------------------------------------------
EMAIL_ADDRESS = "mindscan.alerts@gmail.com"
EMAIL_PASSWORD = "cayy eddy wpat jeok"  # Gmail App Password

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

def send_crisis_email(receiver_email, username, stress_level, emotion):
    try:
        msg = EmailMessage()
        msg["Subject"] = "🚨 MindScan CRISIS Alert"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = receiver_email

        msg.set_content(
            f"""
🚨 MINDSCAN CRISIS ALERT 🚨

User ID: {username}
Stress Level: {stress_level}/10
Detected State: {emotion}

Please check on the user as soon as possible.
(This is an automated alert from MindScan)
"""
        )

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        return True

    except Exception as e:
        print("Email error:", e)
        return False
