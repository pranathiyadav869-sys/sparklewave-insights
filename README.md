# SparkleWave Insights — FMCG Business Insights Assistant

An AI-powered conversational analytics assistant for a beverages FMCG
company. Business users ask questions in plain English ("Which promotion
generated the highest revenue?", "Compare North and South sales") and get
back a written business summary, a chart, and the underlying data table —
all generated automatically from a natural-language question via Gemini +
LangChain + SQLite + Plotly.

## What this is

- A synthetic but realistic FMCG dataset (20 products, 30 stores, 24 weeks
  of weekly sales/promotions and inventory data) with deliberately
  engineered business patterns: seasonality, regional preferences, fast vs.
  slow-moving SKUs, promotion effectiveness differences, and stockouts
  caused by demand spikes outrunning replenishment.
- A SQLite database built from that data with a clean star-schema-like
  structure (two dimension tables, two fact tables, one convenience view).
- An LLM agent that converts a natural-language question into SQL, runs it
  safely against the database, summarizes the result in business language,
  and picks an appropriate chart.
- A Streamlit chat interface that ties it all together.

## Project structure

```
fmcg-insights-assistant/
│
├── app.py                  # Streamlit application (chat UI, history, charts)
├── generate_dataset.py     # Synthetic data generator (run first)
├── database.py             # SQLite schema builder + safe query execution layer
├── llm_agent.py             # LangChain + Gemini: NL -> SQL -> summary -> chart
├── schema.sql               # Raw SQL DDL (tables, keys, indexes, view)
├── requirements.txt        # Python dependencies
├── .env.example             # Template for required environment variables
├── .gitignore
├── README.md                 # This file
├── data/                     # Generated CSVs (created by generate_dataset.py)
│   ├── product_master.csv
│   ├── store_master.csv
│   ├── sales_promotions.csv
│   └── inventory.csv
└── docs/                      # Architecture, assessment write-up, presentation outline
    └── ASSESSMENT_SUBMISSION.md
```

### File-by-file explanation

**`generate_dataset.py`**
Generates the four source CSVs from scratch using `pandas`, `numpy`, and
`random` with a fixed seed for reproducibility. Encodes the business logic
described in the assessment (seasonality, regional affinity, promotion
effectiveness, stockout mechanics) and runs a validation suite (row counts,
referential integrity, inventory-equation balance, stockout-flag
consistency) before writing the files. Must be run before anything else.

**`schema.sql`**
Pure SQL DDL: `CREATE TABLE` statements for `product_master`, `store_master`,
`sales_promotions`, and `inventory`, with primary keys, foreign keys, `CHECK`
constraints, indexes on the columns most queries will filter/group by, and a
`vw_sales_full` view that pre-joins everything for convenience.

**`database.py`**
Builds the SQLite file (`fmcg_insights.db`) from `schema.sql` and loads the
four CSVs into it. Also exposes `execute_readonly_query()`, the single
chokepoint through which all LLM-generated SQL must pass — it rejects
anything that isn't a single `SELECT`/`WITH` statement, blocking
INSERT/UPDATE/DELETE/DROP and multi-statement injection attempts. Also
exposes `get_schema_description()`, the schema text injected into the
LLM's prompt so it always knows exactly what it can query.

**`llm_agent.py`**
The reasoning core. Four steps per question: (1) Gemini converts the
question into SQL using the schema description as grounding; (2)
`database.execute_readonly_query()` runs it safely, with one automatic
repair attempt if the first query errors; (3) Gemini is called a second
time to turn the resulting data into a short, plain-language business
summary; (4) a deterministic, rule-based function picks a Plotly chart type
from the shape of the result (no LLM involved in this step — keeps charts
fast and free of hallucinated plotting code).

**`app.py`**
The Streamlit front end: chat interface, sidebar with sample questions and
query history, database/API-key status indicators, a "rebuild database"
button, and rendering for the summary + chart + expandable data table + SQL
transparency panel for every answer.

**`data/`**
Output folder for the four generated CSVs. Treated as a build artifact —
regenerate any time by re-running `generate_dataset.py`.

**`docs/ASSESSMENT_SUBMISSION.md`**
The full architecture write-up, business-logic explanation, deployment
guide, assessment submission answers, and 10-slide presentation outline —
everything from Parts 1, 3, 9, 10, and 11 of the assessment brief.

## Setup (local)

```bash
# 1. Clone / unzip the project, then create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the synthetic dataset
python generate_dataset.py

# 4. Build the SQLite database from the dataset
python database.py

# 5. Set your Gemini API key
cp .env.example .env
# then edit .env and paste your key from https://aistudio.google.com/app/apikey

# 6. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Re-generating data with different parameters

All the "business logic" knobs (seasonality curves, regional affinities,
promotion lift ranges, stockout likelihood) live as clearly commented
constants near the top of `generate_dataset.py` — change them and re-run the
script (then re-run `database.py`) to produce a different-but-still-coherent
dataset.

## Deployment

See `docs/ASSESSMENT_SUBMISSION.md` → Part 9 for the full Streamlit
Community Cloud deployment walkthrough, including how to set the
`GOOGLE_API_KEY` secret.
