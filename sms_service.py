from twilio.rest import Client

# -------------------------------------------------
# TWILIO FREE TRIAL CONFIG
# -------------------------------------------------
ACCOUNT_SID = "YOUR_TWILIO_ACCOUNT_SID"
AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN"
FROM_NUMBER = "+YOUR_TWILIO_PHONE_NUMBER"

client = Client(ACCOUNT_SID, AUTH_TOKEN)

# -------------------------------------------------
# SEND SMS ALERT (THIS NAME MUST MATCH)
# -------------------------------------------------
def send_sms_alert(to_number, username, stress_level, emotion):
    try:
        client.messages.create(
            body=f"""
🚨 MINDSCAN CRISIS ALERT 🚨

User ID: {username}
Stress Level: {stress_level}/10
Detected State: {emotion}

Please check on the user immediately.
""",
            from_=FROM_NUMBER,
            to=to_number
        )
        return True
    except Exception as e:
        print("SMS error:", e)
        return False
