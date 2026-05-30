"""
FakeGuard — Streamlit Dashboard
Tabs: Analyze Article | Analyze URL | Model Comparison | Analytics
"""
from typing import Dict, List, Optional
import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="FakeGuard — Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.fake-badge  { display:inline-block; background:#dc2626; color:#fff;
               padding:6px 18px; border-radius:20px; font-weight:600; font-size:1.05rem; }
.real-badge  { display:inline-block; background:#16a34a; color:#fff;
               padding:6px 18px; border-radius:20px; font-weight:600; font-size:1.05rem; }
.warn-box    { background:#fef9c3; border-left:4px solid #ca8a04;
               padding:0.6rem 0.9rem; border-radius:4px; margin:0.5rem 0; font-size:0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ── API helpers ────────────────────────────────────────────────────────────────

def _get(path: str, **kwargs) -> Optional[Dict]:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5, **kwargs)
        return r.json() if r.ok else None
    except Exception:
        return None


def _post(path: str, json: Dict, params: Optional[Dict] = None) -> Optional[Dict]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=json, params=params, timeout=20)
        if r.ok:
            return r.json()
        err = r.json().get("detail", "Unknown error")
        st.error(f"API {r.status_code}: {err}")
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot reach the API. "
            "Start both services with: `python start_all.py`"
        )
    except Exception as exc:
        st.error(f"Request failed: {exc}")
    return None


def api_healthy() -> bool:
    r = _get("/health")
    return bool(r and r.get("status") == "healthy")


def available_models() -> List[str]:
    r = _get("/models")
    return r.get("available_models", []) if r else []


# ── Result renderer ────────────────────────────────────────────────────────────

def render_result(result: Dict) -> None:
    label      = result.get("label", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    probs      = result.get("probabilities", {})
    model      = result.get("model_used", "N/A")
    warning    = result.get("warning")

    col_badge, col_chart = st.columns([1, 2])

    with col_badge:
        badge_cls = "fake-badge" if label == "FAKE" else "real-badge"
        icon = "❌" if label == "FAKE" else "✅"
        st.markdown(
            f'<span class="{badge_cls}">{icon} {label}</span>',
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.caption(
            f"Model: `{model}` · Text length: {result.get('input_text_length', '?')} chars"
        )

    with col_chart:
        fig = go.Figure(
            go.Bar(
                x=[probs.get("FAKE", 0), probs.get("REAL", 0)],
                y=["FAKE", "REAL"],
                orientation="h",
                marker_color=["#dc2626", "#16a34a"],
                text=[
                    f"{v * 100:.1f}%"
                    for v in [probs.get("FAKE", 0), probs.get("REAL", 0)]
                ],
                textposition="inside",
                insidetextanchor="middle",
            )
        )
        fig.update_layout(
            title="Probability distribution",
            xaxis=dict(range=[0, 1], title="Probability"),
            height=180,
            margin=dict(l=0, r=0, t=36, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    if warning:
        st.markdown(
            f'<div class="warn-box">⚠️ {warning}</div>', unsafe_allow_html=True
        )


def _record_history(result: Dict, source: str = "text") -> None:
    """Append a prediction to the session-state history log."""
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(
        {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "source": source,
            "label": result.get("label"),
            "confidence": f"{result.get('confidence', 0) * 100:.1f}%",
            "model": result.get("model_used"),
        }
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 FakeGuard")
    st.caption("AI-Powered Fake News Detection")
    st.divider()

    healthy = api_healthy()
    if healthy:
        st.success("✅ API online")
    else:
        st.error("❌ API offline")
        st.info(
            "Start both services:\n\n"
            "```\npython start_all.py\n```\n\n"
            "Or open two terminals:\n"
            "1. `uvicorn src.app.main:app --port 8000`\n"
            "2. `streamlit run src/app/streamlit_app.py`"
        )

    models = available_models()
    selected_model = (
        st.selectbox("Model", models, index=0)
        if models
        else "logistic_regression"
    )
    if not models:
        st.info(
            "No trained models found.\n"
            "Run:  `python scripts/run_training.py`"
        )

    st.divider()
    st.markdown("**Dataset:** ISOT Fake News (44,898 articles)")
    st.markdown("**Models:** LR · RF · GB · SVC")
    # Only show docs link when API is online — avoids confusing 'page not found'
    if healthy:
        st.markdown("[📖 API docs ↗](http://localhost:8000/docs)")
    else:
        st.caption("API docs: http://localhost:8000/docs (start API first)")


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("## 🔍 FakeGuard — Fake News Detection System")
st.caption("Classify news articles as Real or Fake using machine learning.")

tab_article, tab_url, tab_compare, tab_analytics = st.tabs(
    ["📝 Analyze Article", "🌐 Analyze URL", "📊 Model Comparison", "📈 Analytics"]
)


# ── Tab 1: Article ─────────────────────────────────────────────────────────────
with tab_article:
    st.subheader("Paste article text for instant classification")

    with st.form("article_form"):
        headline = st.text_input(
            "Headline (optional)", placeholder="Enter article headline…"
        )
        body = st.text_area(
            "Article body *",
            placeholder="Paste the full news article text here…",
            height=240,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            submitted = st.form_submit_button(
                "🔎 Detect", type="primary", use_container_width=True
            )
        with col_b:
            compare_all = st.form_submit_button(
                "⚖️ Compare all models", use_container_width=True
            )

    if (submitted or compare_all) and not body.strip():
        st.warning("Please paste article text before submitting.")

    elif submitted and body.strip():
        with st.spinner("Classifying…"):
            result = _post(
                "/predict",
                {"title": headline, "text": body},
                {"model_name": selected_model},
            )
        if result:
            st.divider()
            render_result(result)
            _record_history(result, source="text")

    elif compare_all and body.strip():
        with st.spinner("Running all models…"):
            results = _post("/predict/all-models", {"title": headline, "text": body})
        if results:
            st.divider()
            st.subheader("All-model comparison")
            rows = [
                {
                    "Model": name.replace("_", " ").title(),
                    "Label": r.get("label"),
                    "Confidence": r.get("confidence"),
                    "P(FAKE)": r.get("probabilities", {}).get("FAKE", 0),
                    "P(REAL)": r.get("probabilities", {}).get("REAL", 0),
                }
                for name, r in results.items()
            ]
            df_cmp = pd.DataFrame(rows)
            fig = px.bar(
                df_cmp,
                x="Model",
                y=["P(FAKE)", "P(REAL)"],
                barmode="stack",
                color_discrete_map={"P(FAKE)": "#dc2626", "P(REAL)": "#16a34a"},
                title="Probability per model",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_cmp, use_container_width=True)
            # Record the first model's result for history
            first = next(iter(results.values()), None)
            if first:
                _record_history(first, source="all-models")


# ── Tab 2: URL ─────────────────────────────────────────────────────────────────
with tab_url:
    st.subheader("Analyse a live news URL")
    st.caption(
        "The system scrapes the article body automatically. "
        "Paywalled or bot-protected pages fail with a clear 422 error."
    )

    with st.form("url_form"):
        url_input = st.text_input(
            "Article URL", placeholder="https://www.bbc.com/news/…"
        )
        url_submit = st.form_submit_button(
            "🌐 Analyse URL", type="primary", use_container_width=True
        )

    if url_submit and not url_input.strip():
        st.warning("Please enter a URL.")
    elif url_submit:
        with st.spinner(f"Fetching and classifying: {url_input}…"):
            result = _post(
                "/predict/url",
                {"url": url_input},
                {"model_name": selected_model},
            )
        if result:
            st.divider()
            if result.get("extracted_title"):
                st.markdown(f"**Extracted title:** {result['extracted_title']}")
            render_result(result)
            _record_history(result, source="url")


# ── Tab 3: Model Comparison ────────────────────────────────────────────────────
with tab_compare:
    st.subheader("Model performance on test split (20% holdout)")

    metrics_data = _get("/metrics")

    if metrics_data:
        metric_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        rows = [
            {
                "Model": name.replace("_", " ").title(),
                **{k: m.get(k, 0) for k in metric_cols},
            }
            for name, m in metrics_data.items()
        ]
        df_m = pd.DataFrame(rows)

        col_radar, col_bar = st.columns(2)

        with col_radar:
            fig_radar = go.Figure()
            labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
            for _, row in df_m.iterrows():
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=[row[c] for c in metric_cols],
                        theta=labels,
                        fill="toself",
                        name=row["Model"],
                    )
                )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0.8, 1.0])),
                title="Radar — all metrics",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_bar:
            df_melt = df_m.melt(id_vars="Model", var_name="Metric", value_name="Score")
            fig_bar = px.bar(
                df_melt,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                range_y=[0.8, 1.0],
                title="Metric breakdown by model",
                height=420,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            df_m.rename(columns={"f1_score": "F1", "roc_auc": "ROC-AUC"})
            .style.highlight_max(
                subset=["accuracy", "precision", "recall", "F1", "ROC-AUC"],
                color="#bbf7d0",
            )
            .format(
                {c: "{:.4f}" for c in ["accuracy", "precision", "recall", "F1", "ROC-AUC"]}
            ),
            use_container_width=True,
        )
    else:
        st.info(
            "No metrics found. "
            "Train the models first: `python scripts/run_training.py`"
        )


# ── Tab 4: Analytics ───────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader("System design & pipeline overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "**Data Ingestion**\n"
            "- ISOT Fake News Dataset\n"
            "- 44,898 total articles\n"
            "- 23,481 fake · 21,417 real\n"
            "- Stratified 80/20 split"
        )
    with col2:
        st.markdown(
            "**Preprocessing**\n"
            "- Lowercase + URL/HTML strip\n"
            "- NLTK lemmatisation\n"
            "- Stopword removal\n"
            "- TF-IDF (50k features, bigrams)"
        )
    with col3:
        st.markdown(
            "**Classification**\n"
            "- Logistic Regression (default)\n"
            "- Random Forest (ensemble)\n"
            "- Gradient Boosting (accuracy)\n"
            "- Linear SVC (speed + scale)"
        )

    st.divider()
    st.markdown("**Pipeline stages**")
    pipeline_df = pd.DataFrame(
        {
            "Stage": [
                "Input", "Validation", "Cleaning",
                "Vectorisation", "Classification", "Response",
            ],
            "Tool": [
                "Text / URL / Batch", "Pydantic v2", "NLTK · regex",
                "TF-IDF (scikit-learn)", "4 ML models", "FastAPI · Streamlit",
            ],
            "Output": [
                "Raw article string",
                "Validated schema",
                "Normalised token list",
                "Sparse matrix (50k cols)",
                "FAKE / REAL + probabilities",
                "JSON · Dashboard",
            ],
        }
    )
    st.table(pipeline_df)

    st.divider()
    st.markdown("**Prediction history** (this session)")

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)

        col_dl, col_clr = st.columns([3, 1])
        with col_clr:
            if st.button("🗑 Clear history"):
                st.session_state.history = []
                st.rerun()
    else:
        st.caption("No predictions yet in this session. Run a detection above.")
