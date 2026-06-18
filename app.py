"""
app.py
======
Streamlit front-end for the FMCG Beverages Business Insights Assistant.

Run locally:
    streamlit run app.py

Requires:
    - fmcg_insights.db to exist (run `python database.py` once first, which
      itself requires `python generate_dataset.py` to have been run first).
    - GOOGLE_API_KEY (or GEMINI_API_KEY) set as an environment variable or
      in Streamlit secrets (see PART 9 deployment guide in docs/).
"""

import os
import time
import traceback
from datetime import datetime

import pandas as pd
import streamlit as st

from database import build_database, DB_PATH
import llm_agent

# ---------------------------------------------------------------------------
# PAGE CONFIG  (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SparkleWave Insights | FMCG Business Assistant",
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME / STYLING
# ---------------------------------------------------------------------------
# A deliberate beverages-brand palette rather than default Streamlit blue:
# deep teal (trust, hydration) + citrus orange accent (energy, FMCG shelf
# color) + warm off-white background. Applied via a small CSS injection.
PRIMARY_TEAL = "#0E5E5E"
ACCENT_ORANGE = "#F2762E"
BG_CREAM = "#FAF7F2"
TEXT_DARK = "#1E2A2A"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {BG_CREAM};
    }}
    h1, h2, h3 {{
        color: {PRIMARY_TEAL};
        font-family: 'Helvetica Neue', sans-serif;
    }}
    .hero-banner {{
        background: linear-gradient(135deg, {PRIMARY_TEAL} 0%, #0A4747 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.2rem;
    }}
    .hero-banner h1 {{
        color: white !important;
        margin-bottom: 0.2rem;
    }}
    .hero-banner p {{
        color: #E6F2F0;
        margin: 0;
        font-size: 0.95rem;
    }}
    .stChatMessage {{
        border-radius: 12px;
    }}
    .metric-card {{
        background-color: white;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        border-left: 4px solid {ACCENT_ORANGE};
    }}
    div[data-testid="stSidebar"] {{
        background-color: #F1ECE3;
    }}
    .sql-box {{
        background-color: #14201F;
        color: #9FE6D0;
        padding: 0.8rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        overflow-x: auto;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SAMPLE QUESTIONS (shown as quick-pick buttons)
# ---------------------------------------------------------------------------
SAMPLE_QUESTIONS = [
    "Which promotion generated the highest revenue?",
    "Compare North and South sales.",
    "Which products had stockouts?",
    "Which stores underperformed?",
    "What was the impact of BOGO promotions?",
    "Show top performing products.",
]

# ---------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: question, response, timestamp

if "db_ready" not in st.session_state:
    st.session_state.db_ready = os.path.exists(DB_PATH)

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🥤 SparkleWave Insights")
    st.caption("AI-powered FMCG Business Insights Assistant")

    st.divider()

    st.markdown("**Database status**")
    if st.session_state.db_ready:
        st.success("Database connected", icon="✅")
    else:
        st.warning("Database not found", icon="⚠️")

    if st.button("🔄 Rebuild database from CSVs", use_container_width=True):
        with st.spinner("Rebuilding database from data/ CSV files..."):
            try:
                build_database(verbose=False)
                st.session_state.db_ready = True
                st.success("Database rebuilt successfully.")
            except Exception as e:
                st.error(f"Failed to rebuild database: {e}")

    st.divider()

    api_key_set = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    st.markdown("**Gemini API status**")
    if api_key_set:
        st.success("API key detected", icon="✅")
    else:
        st.error("No API key found", icon="🚫")
        st.caption(
            "Set GOOGLE_API_KEY as an environment variable, in a local .env "
            "file, or in Streamlit Cloud's app secrets."
        )

    st.divider()

    st.markdown("**Try a sample question**")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    st.markdown("**Query history**")
    if st.session_state.chat_history:
        for i, entry in enumerate(reversed(st.session_state.chat_history[-10:])):
            st.caption(f"{entry['timestamp']} — {entry['question'][:48]}"
                       f"{'...' if len(entry['question']) > 48 else ''}")
    else:
        st.caption("No questions asked yet this session.")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ---------------------------------------------------------------------------
# HERO BANNER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>SparkleWave Insights</h1>
        <p>Ask questions about promotions, inventory, regional sales, and product
        performance in plain English — get back a business summary, the data,
        and a chart, all in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# GUARD: database must exist before allowing questions
# ---------------------------------------------------------------------------
if not st.session_state.db_ready:
    st.error(
        "The database hasn't been built yet. Run `python generate_dataset.py` "
        "and then `python database.py` in your terminal, or use the **Rebuild "
        "database** button in the sidebar (requires the CSVs in `data/` to "
        "already exist)."
    )
    st.stop()


# ---------------------------------------------------------------------------
# RENDER A SINGLE ASSISTANT RESPONSE (reused for history + new answers)
# ---------------------------------------------------------------------------
def render_response(entry: dict):
    response = entry["response"]

    if response.error:
        st.error(f"Something went wrong answering this question: {response.error}")
        if response.sql_query:
            with st.expander("🔍 SQL that was attempted"):
                st.markdown(f'<div class="sql-box">{response.sql_query}</div>',
                            unsafe_allow_html=True)
        return

    # Business summary
    st.markdown(response.summary)

    # Chart (if one was built)
    if response.chart is not None:
        st.plotly_chart(response.chart, use_container_width=True,
                         key=f"chart_{entry['id']}")

    # Data table
    if response.dataframe is not None and not response.dataframe.empty:
        with st.expander(f"📊 View underlying data ({len(response.dataframe)} rows)"):
            st.dataframe(response.dataframe, use_container_width=True)
            csv_data = response.dataframe.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download as CSV",
                data=csv_data,
                file_name=f"insight_{entry['id']}.csv",
                mime="text/csv",
                key=f"download_{entry['id']}",
            )

    # SQL transparency panel
    with st.expander("🔍 SQL query used"):
        if response.repaired:
            st.caption("Note: the initial query needed one automatic correction.")
        st.markdown(f'<div class="sql-box">{response.sql_query}</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# RENDER EXISTING CHAT HISTORY
# ---------------------------------------------------------------------------
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        render_response(entry)


# ---------------------------------------------------------------------------
# HANDLE NEW QUESTION (from chat input OR sidebar sample-question buttons)
# ---------------------------------------------------------------------------
def handle_question(question: str):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Interpreting your question and querying the data..."):
            try:
                response = llm_agent.ask_assistant(question)
            except Exception as e:
                # Catch-all so a single bad turn never crashes the whole app.
                class _ErrShim:
                    pass
                response = llm_agent.AssistantResponse(question=question)
                response.error = f"Unexpected error: {e}"
                if os.environ.get("DEBUG"):
                    st.code(traceback.format_exc())

        entry = {
            "id": len(st.session_state.chat_history),
            "question": question,
            "response": response,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        render_response(entry)

    st.session_state.chat_history.append(entry)


chat_input = st.chat_input("Ask about promotions, sales, inventory, or stockouts...")

if st.session_state.pending_question:
    handle_question(st.session_state.pending_question)
    st.session_state.pending_question = None
elif chat_input:
    handle_question(chat_input)


# ---------------------------------------------------------------------------
# EMPTY STATE (only shown if no chat history yet and no input given)
# ---------------------------------------------------------------------------
if not st.session_state.chat_history and not chat_input:
    st.info(
        "👋 Try one of the sample questions in the sidebar, or type your own "
        "question below — for example: *\"What was the impact of BOGO "
        "promotions in the South region?\"*"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><b>📈 Sales & Promotions</b><br>'
            'Revenue, units sold, promo lift by type, region, or product</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><b>📦 Inventory & Stockouts</b><br>'
            'Stock levels, stockout frequency, fast vs. slow movers</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><b>🗺️ Regional Comparisons</b><br>'
            'North vs South vs East vs West performance breakdowns</div>',
            unsafe_allow_html=True,
        )
