"""
Phase 4: The frontend. A Streamlit app that talks to the FastAPI backend.

Run locally (API must be running first):
    streamlit run frontend/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

EXAMPLES = {
    "Choose an example...": "",
    "Sensational headline": (
        "BREAKING: You won't BELIEVE what the government has been hiding from you!!! "
        "Insiders EXPOSE the shocking truth the mainstream media refuses to report. Share before it's deleted!"
    ),
    "Wire-style report": (
        "The central bank held its benchmark interest rate steady on Tuesday, the monetary policy "
        "committee said in a statement, citing easing inflation and stable foreign exchange reserves."
    ),
}

st.set_page_config(page_title="Fake News Detector", page_icon="🔎", layout="centered")

st.markdown(
    """
    <style>
      .result-card {padding: 1.2rem 1.4rem; border-radius: 12px; margin: 1rem 0;}
      .fake {background: rgba(220, 53, 69, 0.12); border-left: 6px solid #dc3545;}
      .real {background: rgba(25, 135, 84, 0.12); border-left: 6px solid #198754;}
      .result-title {font-size: 1.5rem; font-weight: 700; margin: 0;}
      .result-sub {opacity: 0.8; margin: 0.2rem 0 0 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔎 Fake News Detector")
st.caption("Paste a headline or article. The model looks for writing patterns common in misinformation.")


# --- Sidebar: live monitoring numbers -----------------------------------------
with st.sidebar:
    st.header("📊 Live model stats")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        stats = requests.get(f"{API_URL}/stats", timeout=5).json()
        st.success(f"API online · model {health.get('model_version')}")
        st.metric("Predictions served", stats["total_predictions"])
        if stats["fake_ratio"] is not None:
            st.metric("Flagged as fake", f"{stats['fake_ratio'] * 100:.0f}%")
            st.metric("Average confidence", f"{stats['avg_confidence'] * 100:.0f}%")
            st.metric("Average latency", f"{stats['avg_latency_ms']:.1f} ms")
        if stats["feedback_accuracy"] is not None:
            st.metric(
                "Accuracy from user feedback",
                f"{stats['feedback_accuracy'] * 100:.0f}%",
                help=f"Based on {stats['feedback_count']} 👍/👎 responses",
            )
    except requests.RequestException:
        st.error(f"Cannot reach the API at {API_URL}")
    st.divider()
    st.caption("Built with scikit-learn · FastAPI · Streamlit · Docker")


# --- Main input -----------------------------------------------------------------
example = st.selectbox("Try an example", list(EXAMPLES.keys()))
text = st.text_area(
    "Text to check",
    value=EXAMPLES[example],
    height=180,
    placeholder="Paste a headline or news article here (at least 20 characters)...",
)

if st.button("Check", type="primary", width="stretch"):
    if len(text.strip()) < 20:
        st.warning("Please enter at least 20 characters.")
    else:
        with st.spinner("Analysing..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=15)
                resp.raise_for_status()
                st.session_state["result"] = resp.json()
                st.session_state["feedback_sent"] = False
            except requests.RequestException as e:
                st.error(f"Prediction failed: {e}")

result = st.session_state.get("result")
if result:
    is_fake = result["label"] == "fake"
    css = "fake" if is_fake else "real"
    title = "⚠️ Likely misinformation-style" if is_fake else "✅ Looks like normal news writing"
    st.markdown(
        f"""
        <div class="result-card {css}">
          <p class="result-title">{title}</p>
          <p class="result-sub">Confidence: {result['confidence'] * 100:.1f}% · model {result['model_version']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(result["prob_fake"], text=f"Probability of fake: {result['prob_fake'] * 100:.1f}%")
    st.info(result["disclaimer"])

    # --- Feedback: this is how we measure real-world accuracy --------------------
    if not st.session_state.get("feedback_sent"):
        st.write("Was this prediction correct?")
        c1, c2, _ = st.columns([1, 1, 4])
        for col, correct, label in [(c1, True, "👍 Yes"), (c2, False, "👎 No")]:
            if col.button(label):
                try:
                    requests.post(
                        f"{API_URL}/feedback",
                        json={"request_id": result["request_id"], "correct": correct},
                        timeout=5,
                    ).raise_for_status()
                    st.session_state["feedback_sent"] = True
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"Could not save feedback: {e}")
    else:
        st.success("Thanks for the feedback! It helps us monitor the model.")
