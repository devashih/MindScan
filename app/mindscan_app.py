# --- keep these first so Transformers never pulls TensorFlow by accident ---
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# --------------------------------------------------------------------------

import pandas as pd
import streamlit as st
from transformers import pipeline
from io import BytesIO
from PIL import Image

# Optional image emotion (install: fer, opencv-python-headless, tensorflow-cpu==2.13.*, keras<3)
FER_AVAILABLE = True
try:
    from fer import FER
    import numpy as np
except Exception:
    FER_AVAILABLE = False

# Local DB helpers (make sure app/db.py exists)
from db import add_user, check_user, save_entry, fetch_entries

# ──────────────────────────────────────────────────────────────────────────────
# Page setup + dark card theme
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MindScan – Secure Journal", page_icon="🧠", layout="wide")

# Global Altair theme to match your screenshot
def _register_ms_altair_theme():
    import altair as alt
    LINE = "#8AD0FF"   # cyan line
    BAR  = "#A9D5FF"   # pastel blue bars
    TXT  = "#E6E6E6"   # labels
    GRID = "#2A3A4A"   # subtle grid

    alt.themes.register("mindscan_dark", lambda: {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent", "continuousWidth": 400, "continuousHeight": 300},
            "axis": {
                "labelColor": TXT, "titleColor": TXT,
                "grid": True, "gridColor": GRID, "domainColor": GRID, "tickColor": GRID
            },
            "legend": {"labelColor": TXT, "titleColor": TXT},
            "range": {"category": [LINE, BAR]}
        }
    })
    alt.themes.enable("mindscan_dark")

try:
    _register_ms_altair_theme()
except Exception:
    pass

# CSS for cards / layout (dark background behind charts)
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1100px;}
h1, h2, h3 { font-weight: 700 !important; letter-spacing: .2px; }
.ms-card{
  padding:18px 18px 14px; border-radius:16px;
  background:#0F141B;             /* screenshot-style card */
  border:1px solid rgba(255,255,255,.06);
}
.ms-pad { padding:.4rem .6rem; }
.stTextArea textarea, .stTextInput input { border-radius:12px !important; }
button[kind="primary"] { border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🧠 MindScan – Secure Journal")

# ──────────────────────────────────────────────────────────────────────────────
# NLP (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

@st.cache_resource
def load_emotion():
    return pipeline("text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True)

sentiment_pipe = load_sentiment()
emotion_pipe = load_emotion()

def analyze_text(text: str):
    if not text or not text.strip():
        return None, None
    s = sentiment_pipe(text)[0]
    sentiment = {"LABEL_0": -1, "LABEL_1": 0, "LABEL_2": 1}.get(s["label"], 0)
    e = emotion_pipe(text)[0]
    emotion = max(e, key=lambda x: x["score"])["label"]
    return float(sentiment), str(emotion)

# ──────────────────────────────────────────────────────────────────────────────
# Image emotion (optional)
# ──────────────────────────────────────────────────────────────────────────────
EMO_MAP_TO_SENT = {
    "happy": 1.0, "neutral": 0.0, "sad": -1.0,
    "angry": -1.0, "fear": -1.0, "disgust": -1.0, "surprise": 0.2,
}

def analyze_image(img: Image.Image):
    if not FER_AVAILABLE:
        return None, None
    try:
        arr = np.array(img.convert("RGB"))
        detector = FER(mtcnn=True)
        result = detector.top_emotion(arr)
        if result is None:
            return None, None
        emo, _ = result
        emo = (emo or "").lower()
        return EMO_MAP_TO_SENT.get(emo, 0.0), emo
    except Exception:
        return None, None

def combine_modalities(text_sent, text_emo, img_sent, img_emo):
    if text_sent is None and img_sent is None:
        return 0.0, "neutral"
    if img_sent is None:
        return float(text_sent), text_emo or "neutral"
    if text_sent is None:
        return float(img_sent), img_emo or "neutral"
    final_sent = 0.7 * float(text_sent) + 0.3 * float(img_sent)
    final_emo = text_emo or img_emo or "neutral"
    return float(final_sent), final_emo

# ──────────────────────────────────────────────────────────────────────────────
# Charts: screenshot-style dark theme (Altair) with matplotlib fallback
# ──────────────────────────────────────────────────────────────────────────────
def render_trend_charts(df: pd.DataFrame):
    """
    Screenshot-style:
      Top: sentiment line over time (−1..1), cyan line, subtle grid, transparent bg.
      Bottom: emotion frequency bars, pastel blue, x labels vertical.
    """
    try:
        import altair as alt

        data = df.copy()
        data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce")
        data["sentiment"]  = pd.to_numeric(data["sentiment"], errors="coerce")
        data = data.dropna(subset=["created_at", "sentiment"]).sort_values("created_at")
        if data.empty:
            st.info("Not enough data to render charts yet.")
            return

        # Sentiment line (top)
        line = (
            alt.Chart(data)
            .mark_line(point=False, strokeWidth=2, color="#8AD0FF")
            .encode(
                x=alt.X("created_at:T", title=None,
                        axis=alt.Axis(format="%I %p", labelAngle=0, tickCount=8)),
                y=alt.Y("sentiment:Q", title=None, scale=alt.Scale(domain=[-1, 1])),
                tooltip=[alt.Tooltip("created_at:T", title="Time"),
                         alt.Tooltip("sentiment:Q", title="Sentiment")]
            )
            .properties(height=240)
        )

        # Emotion bars (bottom)
        emo_df = (
            data.assign(emotion=data["emotion"].astype(str))
                .groupby("emotion", as_index=False)
                .size().rename(columns={"size": "count"})
        )
        bars = (
            alt.Chart(emo_df)
            .mark_bar(color="#A9D5FF")
            .encode(
                x=alt.X("emotion:N", sort="-y",
                        axis=alt.Axis(labelAngle=90, title=None)),
                y=alt.Y("count:Q", title=None),
                tooltip=["emotion:N", "count:Q"]
            )
            .properties(height=260)
        )

        # Render inside dark card (charts are transparent)
        st.markdown('<div class="ms-card">', unsafe_allow_html=True)
        st.altair_chart((line & bars), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception:
        # Matplotlib fallback with matching theme
        import matplotlib.pyplot as plt
        LINE = "#8AD0FF"; BAR = "#A9D5FF"; TXT = "#E6E6E6"; GRID = "#2A3A4A"

        tmp = df.copy()
        tmp["created_at"] = pd.to_datetime(tmp["created_at"], errors="coerce")
        tmp["sentiment"]  = pd.to_numeric(tmp["sentiment"], errors="coerce")
        tmp = tmp.dropna(subset=["created_at", "sentiment"]).sort_values("created_at")
        if tmp.empty:
            return

        def _style(ax):
            ax.set_facecolor("none")
            ax.grid(True, color=GRID, alpha=0.7)
            ax.tick_params(colors=TXT)
            for spine in ax.spines.values():
                spine.set_color(GRID)

        # top line
        fig1, ax1 = plt.subplots(facecolor="none")
        _style(ax1)
        ax1.plot(tmp["created_at"].values, tmp["sentiment"].astype(float).values,
                 color=LINE, linewidth=2)
        ax1.set_ylim([-1, 1])
        fig1.patch.set_alpha(0)

        # bottom bars
        counts = tmp["emotion"].astype(str).value_counts()
        fig2, ax2 = plt.subplots(facecolor="none")
        _style(ax2)
        ax2.bar(counts.index.values, counts.values, color=BAR)
        for tick in ax2.get_xticklabels(): tick.set_rotation(90)
        fig2.patch.set_alpha(0)

        # wrap in dark card
        st.markdown('<div class="ms-card">', unsafe_allow_html=True)
        st.pyplot(fig1)
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None

# ──────────────────────────────────────────────────────────────────────────────
# Auth UI (login/signup tabs)
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.user_id:
    tab_login, tab_signup = st.tabs(["🔓 Login", "🔑 Signup"])

    with tab_login:
        st.markdown('<div class="ms-card">', unsafe_allow_html=True)
        lg_user = st.text_input("Username", key="login_user")
        lg_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", use_container_width=True):
            uid = check_user(lg_user, lg_pass)
            if uid:
                st.session_state.user_id = uid
                st.session_state.username = lg_user
                st.success(f"✅ Welcome, {lg_user}!")
            else:
                st.error("⚠ Invalid username or password.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_signup:
        st.markdown('<div class="ms-card">', unsafe_allow_html=True)
        su_user = st.text_input("New username", key="su_user")
        su_pass = st.text_input("Password (min 6 chars)", type="password", key="su_pass")
        if st.button("Create account", use_container_width=True):
            if not su_user or not su_pass or len(su_pass) < 6:
                st.error("Please provide a username and a password of at least 6 characters.")
            else:
                st.success("✅ Account created! Please login.") if add_user(su_user, su_pass) \
                    else st.error("⚠ Username already exists.")
        st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.user_id:
    st.markdown("### 📝 Journal")
    st.markdown('<div class="ms-card">', unsafe_allow_html=True)
    with st.form("journal"):
        text = st.text_area("How are you feeling today?", height=160, placeholder="Type your thoughts…")
        img_file = st.file_uploader("Optional face photo (processed in-memory; not saved)",
                                    type=["png", "jpg", "jpeg"])
        submit = st.form_submit_button("Save & Analyze")
        if submit and (text.strip() or img_file):
            t_sent, t_emo = analyze_text(text) if text.strip() else (None, None)
            i_sent, i_emo = (None, None)
            if img_file:
                try:
                    img = Image.open(BytesIO(img_file.read()))
                    i_sent, i_emo = analyze_image(img)
                    if not FER_AVAILABLE:
                        st.info("Image analysis requires extra packages (fer + TF). Only text was analyzed.")
                except Exception:
                    st.warning("Could not analyze the image. Try a clear, front-facing photo.")
            final_sent, final_emo = combine_modalities(t_sent, t_emo, i_sent, i_emo)
            save_entry(st.session_state.user_id, text.strip(), float(final_sent), str(final_emo))
            st.success(f"✅ Saved! Sentiment={final_sent:.2f}, Emotion={final_emo}")
        elif submit:
            st.error("Please enter text or upload an image.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📊 Mood Trends (Last 7 Days)")
    rows = fetch_entries(st.session_state.user_id, days=7)
    if rows:
        df = pd.DataFrame(rows, columns=["created_at", "sentiment", "emotion", "text"])
        df["created_at"] = pd.to_datetime(df["created_at"])
        render_trend_charts(df)
        st.markdown('<div class="ms-card">', unsafe_allow_html=True)
        st.dataframe(df.sort_values("created_at", ascending=False),
                     use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No entries yet. Add your first journal entry to see trends.")

    st.markdown('<div class="ms-card ms-pad">', unsafe_allow_html=True)
    st.write(f"**User:** `{st.session_state.username}`")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.user_id = None
        st.session_state.username = None
        st.success("Logged out.")
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("Privacy: Entries are stored locally in SQLite. Uploaded images are processed in-memory and never saved.")
