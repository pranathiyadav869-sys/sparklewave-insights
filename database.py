"""
database.py
============
Database setup and access layer for the FMCG Business Insights Assistant.

Responsibilities:
  1. Build a fresh SQLite database from schema.sql.
  2. Load the four CSV files produced by generate_dataset.py into their
     respective tables.
  3. Expose a small, safe query-execution interface that the AI assistant
     (llm_agent.py) and the Streamlit app (app.py) both depend on.

Run directly to (re)build the database from scratch:
    python database.py
"""

import os
import sqlite3
from contextlib import contextmanager

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "fmcg_insights.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

TABLE_CSV_MAP = {
    "product_master": "product_master.csv",
    "store_master": "store_master.csv",
    "sales_promotions": "sales_promotions.csv",
    "inventory": "inventory.csv",
}

# Columns that should NOT be loaded from CSV because the table defines them
# as auto-incrementing surrogate keys.
AUTOINCREMENT_COLUMNS = {
    "sales_promotions": "sales_id",
    "inventory": "inventory_id",
}

# Allow-list of tables/columns the assistant is permitted to query. Used by
# the safety guardrails in execute_readonly_query().
ALLOWED_TABLES = {
    "product_master", "store_master", "sales_promotions", "inventory", "vw_sales_full"
}


def build_database(db_path: str = DB_PATH, schema_path: str = SCHEMA_PATH,
                    data_dir: str = DATA_DIR, verbose: bool = True) -> None:
    """
    Drops and rebuilds the SQLite database file from schema.sql, then loads
    all four CSVs. Safe to re-run at any time (idempotent).
    """
    if os.path.exists(db_path):
        os.remove(db_path)
        if verbose:
            print(f"Removed existing database at {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        with open(schema_path, "r") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
        if verbose:
            print("Schema created successfully.")

        for table, csv_file in TABLE_CSV_MAP.items():
            csv_path = os.path.join(data_dir, csv_file)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Expected data file not found: {csv_path}. "
                    f"Run generate_dataset.py first."
                )
            df = pd.read_csv(csv_path)

            # Fill NaN promotion_type (non-promo rows) with empty string to
            # satisfy the CHECK constraint and keep comparisons simple.
            if "promotion_type" in df.columns:
                df["promotion_type"] = df["promotion_type"].fillna("")

            df.to_sql(table, conn, if_exists="append", index=False)
            if verbose:
                print(f"   Loaded {len(df):,} rows into '{table}' from {csv_file}")

        conn.commit()

        # Quick post-load sanity check
        cur = conn.cursor()
        for table in TABLE_CSV_MAP:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            if verbose:
                print(f"   Verified '{table}' row count in DB: {count:,}")

    finally:
        conn.close()

    if verbose:
        print(f"\nDatabase ready at: {db_path}")


@contextmanager
def get_connection(db_path: str = DB_PATH):
    """Context-managed read connection with foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def execute_readonly_query(sql: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Executes a SQL query and returns the result as a pandas DataFrame.

    Safety guardrails (this function is the single chokepoint through which
    all LLM-generated SQL passes before touching the database):
      * Only SELECT statements are allowed (no INSERT/UPDATE/DELETE/DROP/etc).
      * Only one statement may be submitted (blocks ';...' SQL injection
        chaining).
      * Referenced tables must be in the ALLOWED_TABLES allow-list.

    Raises ValueError if the query fails any guardrail check.
    """
    cleaned = sql.strip().rstrip(";")

    if not cleaned:
        raise ValueError("Empty SQL query.")

    lowered = cleaned.lower()

    if not lowered.startswith("select") and not lowered.startswith("with"):
        raise ValueError(
            "Only SELECT (or WITH ... SELECT) statements are permitted. "
            f"Rejected query: {sql[:200]}"
        )

    forbidden_keywords = [
        "insert ", "update ", "delete ", "drop ", "alter ", "create ",
        "truncate ", "attach ", "pragma ", "replace ", "--", "/*"
    ]
    for kw in forbidden_keywords:
        if kw in lowered:
            raise ValueError(f"Query contains forbidden keyword/pattern: '{kw.strip()}'")

    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not permitted.")

    with get_connection(db_path) as conn:
        try:
            df = pd.read_sql_query(cleaned, conn)
        except Exception as e:
            raise ValueError(f"SQL execution failed: {e}")

    return df


def get_schema_description() -> str:
    """
    Returns a human/LLM-readable description of the database schema,
    including column names, types, and a short note on each table. This is
    injected into the LLM prompt so Gemini knows exactly what it can query.
    """
    return """
TABLE: product_master
  - product_id      TEXT PRIMARY KEY   (e.g. 'P001')
  - product_name    TEXT               (e.g. 'FizzPop Cola 1000ml')
  - brand           TEXT
  - category        TEXT               (one of: Juice, Water, Carbonated, Energy Drink, Dairy)
  - sub_category    TEXT
  - pack_size_ml    INTEGER
  - unit_price      REAL

TABLE: store_master
  - store_id        TEXT PRIMARY KEY   (e.g. 'S001')
  - store_name      TEXT
  - region          TEXT               (one of: North, South, East, West)
  - city            TEXT
  - store_format    TEXT               (one of: Supermarket, Convenience Store, Hypermarket, Kirana/Local Mart)

TABLE: sales_promotions   (grain: one row per week x product x store)
  - sales_id          INTEGER PRIMARY KEY
  - week_start_date   TEXT  (format YYYY-MM-DD, Monday of each week)
  - product_id        TEXT  FOREIGN KEY -> product_master.product_id
  - store_id          TEXT  FOREIGN KEY -> store_master.store_id
  - region             TEXT
  - units_sold        INTEGER
  - revenue           REAL
  - promotion_flag    INTEGER (0 or 1)
  - promotion_type    TEXT  (one of: '', 'Price Cut', 'Bundle', 'BOGO', 'Display Feature')
  - discount_pct      REAL  (0-100)

TABLE: inventory   (grain: one row per week x product x store, same grain as sales_promotions)
  - inventory_id      INTEGER PRIMARY KEY
  - week_start_date   TEXT
  - product_id        TEXT  FOREIGN KEY -> product_master.product_id
  - store_id          TEXT  FOREIGN KEY -> store_master.store_id
  - opening_stock     INTEGER
  - units_received    INTEGER
  - units_sold        INTEGER
  - closing_stock     INTEGER
  - stockout_flag     INTEGER (0 or 1; 1 means demand exceeded available stock that week)

VIEW: vw_sales_full
  A pre-joined view combining sales_promotions + inventory + product_master +
  store_master. Prefer this view for questions that need attributes from
  multiple tables at once (e.g. "stockouts by category", "promo revenue by region").
  Columns: sales_id, week_start_date, product_id, product_name, brand, category,
  sub_category, pack_size_ml, unit_price, store_id, store_name, region, city,
  store_format, units_sold, revenue, promotion_flag, promotion_type, discount_pct,
  opening_stock, units_received, closing_stock, stockout_flag.

NOTES FOR QUERY GENERATION:
  - sales_promotions.units_sold and inventory.units_sold are identical for the
    same (week_start_date, product_id, store_id) - they are kept in both tables
    by design, so you rarely need to join them just to get units_sold.
  - To analyze stockouts together with sales/promo context, use vw_sales_full
    or JOIN sales_promotions to inventory ON (week_start_date, product_id, store_id).
  - Dates are stored as TEXT in 'YYYY-MM-DD' format; SQLite string comparison
    works correctly for this format (lexicographic order = chronological order).
  - "underperforming stores" generally means low SUM(revenue) or low
    SUM(units_sold) relative to peers - sort ascending.
  - promotion_type is an empty string '' (not NULL) when promotion_flag = 0.
"""


if __name__ == "__main__":
    build_database()

    # Smoke test: run one query through the public interface
    print("\n--- Smoke test query ---")
    test_df = execute_readonly_query(
        "SELECT category, SUM(revenue) AS total_revenue "
        "FROM vw_sales_full GROUP BY category ORDER BY total_revenue DESC"
    )
    print(test_df)
