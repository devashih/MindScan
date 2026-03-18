import streamlit as st
import pandas as pd
from transformers import pipeline
from datetime import datetime, timedelta

from database import (
    create_tables,
    register_user,
    login_user,
    get_emergency_email,
    save_entry,
    fetch_user_entries
)

from email_service import send_crisis_email
from environment_analysis import analyze_environment
from face_emotion_analysis import analyze_facial_emotion

# -------------------------------------------------
# PAGE SETUP
# -------------------------------------------------
st.set_page_config(page_title="MindScan", layout="centered")
create_tables()

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = None

# -------------------------------------------------
# LOAD TEXT EMOTION MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

emotion_model = load_model()

# -------------------------------------------------
# TEXT SEVERITY LOGIC
# -------------------------------------------------
def assess_text_severity(text: str):
    text = text.lower()
    if any(k in text for k in ["kill myself", "want to die", "end my life"]):
        return "CRISIS"
    if any(k in text for k in ["hopeless", "overwhelmed", "burnt out"]):
        return "MEDIUM"
    if any(k in text for k in ["tired", "fatigued", "drained"]):
        return "MILD"
    return "NEUTRAL"

# -------------------------------------------------
# ALERT COOLDOWN (24 HOURS)
# -------------------------------------------------
COOLDOWN_HOURS = 24

def can_send_alert():
    if st.session_state.last_alert_time is None:
        return True
    return datetime.now() - st.session_state.last_alert_time >= timedelta(hours=COOLDOWN_HOURS)

# =================================================
# AUTH SECTION
# =================================================
if st.session_state.user_id is None:
    st.title("🧠 MindScan – Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user_id = login_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.session_state.last_alert_time = None
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        emergency_email = st.text_input(
            "Emergency Contact Email",
            placeholder="example@gmail.com"
        )

        if st.button("Signup"):
            if not new_user or not new_pass or not emergency_email:
                st.warning("Please fill all fields")
            elif register_user(new_user, new_pass, emergency_email):
                st.success("Account created! Please login.")
            else:
                st.error("Username already exists")

# =================================================
# MAIN DASHBOARD
# =================================================
else:
    st.title("🧠 MindScan – Dashboard")

    if st.button("Logout"):
        st.session_state.user_id = None
        st.session_state.last_alert_time = None
        st.rerun()

    st.divider()

    # -------------------------------------------------
    # INPUT FORM
    # -------------------------------------------------
    with st.form("entry_form", clear_on_submit=True):
        journal_text = st.text_area(
            "Write how you feel (optional)",
            height=150
        )

        image = st.file_uploader(
            "Upload an image (optional)",
            type=["jpg", "jpeg", "png"]
        )

        submitted = st.form_submit_button("Analyze & Save")

    # -------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------
    if submitted:
        if journal_text.strip() == "" and image is None:
            st.error("Please provide at least text OR an image.")
        else:

            # ---------- TEXT ANALYSIS ----------
            emotion = "neutral"
            confidence = 0.5
            stress = 4

            if journal_text.strip():

                cleaned_text = journal_text.lower().strip()

                results = emotion_model(cleaned_text)
                emotions = results[0]

                top = max(emotions, key=lambda x: x["score"])
                emotion = top["label"]
                confidence = float(top["score"])

                severity = assess_text_severity(cleaned_text)

                if severity == "CRISIS":
                    stress = 9

                elif severity == "MEDIUM":
                    stress = 7

                elif severity == "MILD":
                    stress = 5

                else:
                    emotion_stress_map = {
                        "anger": 6,
                        "sadness": 7,
                        "fear": 7,
                        "disgust": 6,
                        "surprise": 4,
                        "neutral": 3,
                        "joy": 2
                    }

                    stress = emotion_stress_map.get(emotion, 3)

                stress = min(round(stress + confidence * 1.5), 10)

            # ---------- IMAGE ANALYSIS ----------
            if image is not None:
                _, env_risk = analyze_environment(image)
                face_emotion, _, face_risk = analyze_facial_emotion(image)
                stress = min(stress + env_risk + face_risk, 10)
            else:
                face_emotion = "not_detected"

            # ---------- RISK LEVEL ----------
            if stress <= 3:
                risk = "INFO"
            elif stress <= 6:
                risk = "MEDIUM"
            else:
                risk = "CRISIS"

            # ---------- SAVE ENTRY ----------
            save_entry(
                st.session_state.user_id,
                journal_text if journal_text.strip() else "[Image-only entry]",
                emotion,
                confidence,
                stress,
                risk
            )

            # ---------- OUTPUT ----------
            st.success("Analysis complete ✅")
            st.write(f"**Stress Level:** {stress}/10")
            st.write(f"**Risk Level:** {risk}")
            st.write(f"**Text Emotion:** {emotion}")
            st.write(f"**Facial Emotion:** {face_emotion}")

            # ---------- EMAIL ALERT ----------
            if risk == "CRISIS":
                if can_send_alert():
                    receiver = get_emergency_email(st.session_state.user_id)

                    if receiver:
                        sent = send_crisis_email(
                            receiver_email=receiver,
                            username=str(st.session_state.user_id),
                            stress_level=stress,
                            emotion=f"{emotion} / {face_emotion}"
                        )

                        if sent:
                            st.session_state.last_alert_time = datetime.now()
                            st.error("🚨 CRISIS detected! Emergency email sent.")
                        else:
                            st.error("❌ Email sending failed.")
                else:
                    st.warning("🚫 Crisis detected, but alert cooldown is active.")

    st.divider()

    # -------------------------------------------------
    # HISTORY
    # -------------------------------------------------
    st.subheader("📊 Your Mental Health History")

    data = fetch_user_entries(st.session_state.user_id)

    if data:
        df = pd.DataFrame(
            data,
            columns=["Emotion", "Confidence", "Stress", "Risk", "Date"]
        )

        st.dataframe(df, use_container_width=True)

        df["Date"] = pd.to_datetime(df["Date"])
        st.line_chart(df.set_index("Date")["Stress"])
    else:
        st.info("No entries yet.")