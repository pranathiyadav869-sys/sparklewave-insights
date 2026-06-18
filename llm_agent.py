"""
llm_agent.py
============
The AI reasoning layer of the FMCG Business Insights Assistant.

Pipeline (per user question):

    1. NL -> SQL        Gemini (via LangChain) converts the natural language
                         question into a single read-only SQL query, using
                         the schema description from database.py as grounding.
    2. SQL -> DataFrame  database.execute_readonly_query() runs the query
                         against SQLite with safety guardrails.
    3. DataFrame -> Insight   Gemini is called again with the question + the
                         resulting data (as compact JSON/markdown) and asked
                         to write a short business-analyst-style summary.
    4. DataFrame -> Chart    A lightweight rule-based chart-spec chooser picks
                         an appropriate Plotly chart type and builds the
                         figure directly from the DataFrame (no LLM needed for
                         this step - keeps it fast, deterministic, and free of
                         hallucinated chart code).

Design choices worth calling out:
  * We do NOT let the LLM execute SQL itself or call open-ended Python tools.
    It only ever produces text (a SQL string, then a summary string). This
    keeps the "agent" deterministic, auditable, and safe - there is no
    arbitrary code execution risk.
  * SQL generation and summary generation are two separate calls rather than
    one combined call. This keeps each prompt focused, makes failures easier
    to diagnose (a bad SQL query vs. a bad summary are different failure
    modes), and lets us validate/repair the SQL before spending a second LLM
    call on a summary of bad data.
  * If SQL generation fails validation (via database.execute_readonly_query's
    guardrails) or errors at execution time, we attempt ONE automatic repair
    pass by feeding the error back to Gemini, then give up gracefully.
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database import execute_readonly_query, get_schema_description

# ---------------------------------------------------------------------------
# LLM client setup
# ---------------------------------------------------------------------------
# We use LangChain's ChatGoogleGenerativeAI wrapper around the Gemini API.
# The API key is read from the environment (set via .env locally, or via
# Streamlit Cloud's "Secrets" in deployment - see PART 9 deployment guide).

GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-flash")


def _get_llm(temperature: float = 0.0):
    """
    Lazily constructs the LangChain Gemini chat client. Imported lazily
    inside the function (rather than at module load time) so that this
    module can still be imported - e.g. for unit testing the SQL-cleaning
    helpers - in environments where langchain-google-genai isn't installed.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable is not set. "
            "Set it in a .env file locally, or in Streamlit Cloud's app secrets."
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=api_key,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class AssistantResponse:
    question: str
    sql_query: str = ""
    dataframe: Optional[pd.DataFrame] = None
    summary: str = ""
    chart: Optional[go.Figure] = None
    chart_type: str = "none"
    error: Optional[str] = None
    repaired: bool = False  # True if the SQL needed an auto-repair pass


# ---------------------------------------------------------------------------
# STEP 1: Natural language -> SQL
# ---------------------------------------------------------------------------
SQL_SYSTEM_PROMPT = """You are a meticulous SQL analyst working for a beverages FMCG company.
You translate business questions into a SINGLE valid SQLite SELECT query.

DATABASE SCHEMA:
{schema}

RULES (follow all of them exactly):
1. Output ONLY the SQL query. No explanation, no markdown code fences, no comments.
2. The query MUST start with SELECT or WITH.
3. Use only the tables/view described in the schema above.
4. Prefer the `vw_sales_full` view when the question needs columns from more
   than one table (e.g. product category + region + stockout_flag together).
5. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, or ATTACH.
6. Never include a trailing semicolon.
7. Always alias aggregate columns with a clear name (e.g. SUM(revenue) AS total_revenue).
8. For "top N" or "highest/lowest" questions, always include ORDER BY and LIMIT.
9. For "compare X and Y" questions (e.g. compare two regions), filter to just
   those values with WHERE ... IN (...) and GROUP BY the comparison dimension.
10. When the question implies a rate or percentage (e.g. "stockout rate"),
    compute it as AVG(stockout_flag) * 100 rather than just counting rows.
11. If the question is ambiguous about time period, default to the full
    dataset (do not filter by date) unless the user names a specific period.
"""


def _clean_sql(raw_sql: str) -> str:
    """Strips markdown code fences and surrounding whitespace/semicolons that
    LLMs sometimes add even when explicitly told not to."""
    cleaned = raw_sql.strip()
    cleaned = re.sub(r"^```sql\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = cleaned.strip().rstrip(";").strip()
    return cleaned


def generate_sql(question: str) -> str:
    """Calls Gemini to translate a natural-language question into SQL."""
    llm = _get_llm(temperature=0.0)
    schema = get_schema_description()
    system_prompt = SQL_SYSTEM_PROMPT.format(schema=schema)

    messages = [
        ("system", system_prompt),
        ("human", question),
    ]
    response = llm.invoke(messages)
    return _clean_sql(response.content)


def repair_sql(question: str, broken_sql: str, error_message: str) -> str:
    """One-shot repair: feed the failing SQL and the error back to Gemini."""
    llm = _get_llm(temperature=0.0)
    schema = get_schema_description()
    system_prompt = SQL_SYSTEM_PROMPT.format(schema=schema)

    repair_instruction = f"""The previous SQL query you wrote failed.

Original question: {question}

SQL you wrote:
{broken_sql}

Error returned:
{error_message}

Write a corrected SQL query that fixes this error. Follow all the same rules
as before. Output ONLY the corrected SQL query."""

    messages = [
        ("system", system_prompt),
        ("human", repair_instruction),
    ]
    response = llm.invoke(messages)
    return _clean_sql(response.content)


# ---------------------------------------------------------------------------
# STEP 2: SQL -> DataFrame (delegates to database.py; guardrails live there)
# ---------------------------------------------------------------------------
def run_sql(sql_query: str) -> pd.DataFrame:
    return execute_readonly_query(sql_query)


# ---------------------------------------------------------------------------
# STEP 3: DataFrame -> Business summary
# ---------------------------------------------------------------------------
SUMMARY_SYSTEM_PROMPT = """You are a senior FMCG business analyst presenting insights to a busy
brand manager. Write a concise, business-friendly summary (3-5 sentences,
plain prose, no bullet points, no markdown headers) based ONLY on the data
provided. Call out concrete numbers from the data. If the data suggests a
clear business implication (e.g. a promotion type underperforming, a region
lagging, a stockout risk), state it plainly. Do not invent numbers that
aren't in the provided data. Do not mention SQL, databases, or code."""


def summarize_results(question: str, df: pd.DataFrame) -> str:
    """Calls Gemini to turn a result DataFrame into a short business summary."""
    if df.empty:
        return ("No matching data was found for this question. This could mean "
                "the filters were too narrow (e.g. a product, region, or "
                "promotion type that doesn't exist in the dataset), or the "
                "time period had no qualifying records.")

    llm = _get_llm(temperature=0.3)

    # Cap the data sent to the LLM to keep the prompt small and cheap; for
    # large result sets, summarize the shape rather than dumping every row.
    max_rows_for_llm = 30
    if len(df) > max_rows_for_llm:
        data_for_prompt = df.head(max_rows_for_llm).to_markdown(index=False)
        data_note = f"(showing first {max_rows_for_llm} of {len(df)} rows)"
    else:
        data_for_prompt = df.to_markdown(index=False)
        data_note = f"(all {len(df)} rows)"

    human_prompt = f"""Business question: {question}

Query result {data_note}:
{data_for_prompt}

Write the business summary now."""

    messages = [
        ("system", SUMMARY_SYSTEM_PROMPT),
        ("human", human_prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


# ---------------------------------------------------------------------------
# STEP 4: DataFrame -> Plotly chart (deterministic, rule-based - no LLM)
# ---------------------------------------------------------------------------
def build_chart(df: pd.DataFrame, question: str) -> tuple[Optional[go.Figure], str]:
    """
    Chooses and builds a Plotly chart from the result DataFrame using simple,
    explainable heuristics based on column count/types - not an LLM call.
    This keeps chart rendering instant and free of hallucinated chart code.

    Heuristics:
      - 0 rows               -> no chart
      - 1 row, 1-2 numeric cols -> no chart (single value, better as text/metric)
      - 1 categorical + 1 numeric -> bar chart
      - 1 categorical + 2 numeric -> grouped bar chart
      - date/week column present -> line chart (trend)
      - 2 categorical + 1 numeric -> grouped/stacked bar chart
      - otherwise             -> table only (chart_type = 'table')
    """
    if df is None or df.empty:
        return None, "none"

    if len(df) == 1 and len(df.columns) <= 2:
        return None, "none"

    cols = list(df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in cols if c not in numeric_cols]

    date_like_cols = [c for c in non_numeric_cols if "date" in c.lower() or "week" in c.lower()]

    # --- Trend over time ---
    if date_like_cols and numeric_cols:
        date_col = date_like_cols[0]
        df_sorted = df.sort_values(by=date_col)
        if len(numeric_cols) == 1 and len(non_numeric_cols) == 1:
            fig = px.line(df_sorted, x=date_col, y=numeric_cols[0], markers=True,
                           title=question.strip().capitalize())
            return fig, "line"
        elif len(numeric_cols) >= 1 and len(non_numeric_cols) >= 2:
            # Multiple series over time, e.g. region/category breakdown
            series_col = [c for c in non_numeric_cols if c != date_col][0]
            fig = px.line(df_sorted, x=date_col, y=numeric_cols[0], color=series_col,
                           markers=True, title=question.strip().capitalize())
            return fig, "line"

    # --- Single categorical dimension + single numeric measure -> bar ---
    if len(non_numeric_cols) == 1 and len(numeric_cols) == 1:
        cat_col = non_numeric_cols[0]
        num_col = numeric_cols[0]
        df_sorted = df.sort_values(by=num_col, ascending=False)
        fig = px.bar(df_sorted, x=cat_col, y=num_col, title=question.strip().capitalize(),
                      text_auto=".2s")
        fig.update_layout(xaxis_title=cat_col.replace("_", " ").title(),
                           yaxis_title=num_col.replace("_", " ").title())
        return fig, "bar"

    # --- Single categorical dimension + multiple numeric measures -> grouped bar ---
    if len(non_numeric_cols) == 1 and len(numeric_cols) >= 2:
        cat_col = non_numeric_cols[0]
        fig = px.bar(df, x=cat_col, y=numeric_cols, barmode="group",
                      title=question.strip().capitalize())
        return fig, "bar"

    # --- Two categorical dimensions + one numeric measure -> grouped bar ---
    if len(non_numeric_cols) == 2 and len(numeric_cols) == 1:
        cat_col_a, cat_col_b = non_numeric_cols
        num_col = numeric_cols[0]
        fig = px.bar(df, x=cat_col_a, y=num_col, color=cat_col_b, barmode="group",
                      title=question.strip().capitalize())
        return fig, "bar"

    # --- Fallback: no good chart heuristic matched; show as table only ---
    return None, "table"


# ---------------------------------------------------------------------------
# ORCHESTRATOR: ties all 4 steps together
# ---------------------------------------------------------------------------
def ask_assistant(question: str) -> AssistantResponse:
    """
    Main entry point used by the Streamlit app. Runs the full pipeline and
    returns an AssistantResponse with the SQL used, the data, the chart, and
    the natural-language summary - or a populated `error` field if something
    failed even after the repair attempt.
    """
    result = AssistantResponse(question=question)

    # ---- Step 1: NL -> SQL ----
    try:
        sql_query = generate_sql(question)
    except Exception as e:
        result.error = f"Failed to generate SQL from your question: {e}"
        return result

    result.sql_query = sql_query

    # ---- Step 2: SQL -> DataFrame (with one auto-repair attempt) ----
    try:
        df = run_sql(sql_query)
    except ValueError as first_error:
        try:
            repaired_sql = repair_sql(question, sql_query, str(first_error))
            df = run_sql(repaired_sql)
            result.sql_query = repaired_sql
            result.repaired = True
        except Exception as second_error:
            result.error = (
                f"The query could not be executed even after an automatic repair "
                f"attempt. Original error: {first_error}. Repair attempt error: {second_error}"
            )
            return result

    result.dataframe = df

    # ---- Step 3: DataFrame -> Summary ----
    try:
        result.summary = summarize_results(question, df)
    except Exception as e:
        result.summary = (
            "The data was retrieved successfully, but the assistant could not "
            f"generate a written summary ({e}). The raw results are shown below."
        )

    # ---- Step 4: DataFrame -> Chart (deterministic, no LLM) ----
    fig, chart_type = build_chart(df, question)
    result.chart = fig
    result.chart_type = chart_type

    return result


if __name__ == "__main__":
    # Simple manual smoke test (requires GOOGLE_API_KEY to be set).
    import sys
    test_question = sys.argv[1] if len(sys.argv) > 1 else "Which promotion type generated the highest revenue?"
    print(f"Question: {test_question}\n")
    resp = ask_assistant(test_question)
    print("SQL used:\n", resp.sql_query)
    print("\nError:", resp.error)
    print("\nSummary:\n", resp.summary)
    if resp.dataframe is not None:
        print("\nData:\n", resp.dataframe)
    print("\nChart type:", resp.chart_type)
