"""
generate_dataset.py
====================
Synthetic data generator for the FMCG Beverages Business Insights Assistant.

Generates four interlinked CSV files that mimic a real beverages company's
data warehouse extract:

    1. product_master.csv     - 20 SKUs across 5 categories
    2. store_master.csv       - 30 stores across 4 regions
    3. sales_promotions.csv   - 24 weeks x 20 products x 30 stores (weekly grain)
    4. inventory.csv          - 24 weeks x 20 products x 30 stores (weekly grain)

Design goals
------------
The data is not random noise. It is built so that an analyst (or an LLM
querying it) can find genuine, explainable patterns:

  * Some products are structurally fast-moving, others slow-moving.
  * Some regions over-index on certain categories (e.g. South skews
    Carbonated + Juice, North skews Water + Dairy).
  * Promotions have different effectiveness by type (BOGO lifts volume hard
    but erodes margin; Display Feature gives a modest, cheap lift).
  * Stockouts are a *consequence* of demand spikes (often promo-driven)
    outrunning replenishment, not a coin flip.
  * Seasonality follows a beverages-realistic curve (summer peak for cold
    drinks, water and energy drinks; smaller dip in winter).

Run:
    python generate_dataset.py

Output:
    ./data/product_master.csv
    ./data/store_master.csv
    ./data/sales_promotions.csv
    ./data/inventory.csv

All four files are validated at the end (row counts, referential integrity,
no negative stock, no orphan keys) before being considered "released".
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. REPRODUCIBILITY
# ---------------------------------------------------------------------------
# Fixed seeds so the dataset is identical every time it's regenerated. This
# matters for an assessment: graders/reviewers should be able to re-run the
# script and get the exact same numbers you discussed in your write-up.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_WEEKS = 24
START_DATE = datetime(2025, 1, 6)  # a Monday - first week_start_date

# ---------------------------------------------------------------------------
# 1. PRODUCT MASTER
# ---------------------------------------------------------------------------
# 20 beverage SKUs spread across 5 categories. Each product carries a
# "demand_tier" (not exported, used only to drive sales generation) so we can
# deliberately create fast movers and slow movers, and a "category" that
# drives regional and seasonal skew downstream.

CATEGORIES = ["Juice", "Water", "Carbonated", "Energy Drink", "Dairy"]

BRANDS_BY_CATEGORY = {
    "Juice": ["FreshSip", "OrchardPure"],
    "Water": ["AquaZen", "ClearSpring"],
    "Carbonated": ["FizzPop", "BubbleCola"],
    "Energy Drink": ["VoltUp", "RushMax"],
    "Dairy": ["DairyDelight", "CreamyMoo"],
}

SUB_CATEGORY_BY_CATEGORY = {
    "Juice": ["Orange", "Mixed Fruit", "Apple", "Mango"],
    "Water": ["Still", "Sparkling"],
    "Carbonated": ["Cola", "Lemon-Lime", "Orange Soda"],
    "Energy Drink": ["Classic", "Sugar-Free"],
    "Dairy": ["Flavored Milk", "Yogurt Drink"],
}

PACK_SIZES_ML = [200, 250, 330, 500, 750, 1000, 1500]


def build_product_master():
    """
    Build 20 products. Distribution across categories is intentionally
    uneven, mirroring a real beverages portfolio where Carbonated and Water
    dominate SKU count, while Energy Drink and Dairy are smaller, premium
    lines.
    """
    category_allocation = {
        "Carbonated": 5,
        "Water": 5,
        "Juice": 4,
        "Energy Drink": 3,
        "Dairy": 3,
    }
    assert sum(category_allocation.values()) == 20

    products = []
    product_id_counter = 1

    # demand_tier drives baseline weekly units in the sales generator:
    #   "high"   -> fast-moving SKU (e.g. flagship cola, mainstream water)
    #   "medium" -> steady seller
    #   "low"    -> slow-moving / niche SKU (premium or new sub-category)
    for category, count in category_allocation.items():
        brands = BRANDS_BY_CATEGORY[category]
        sub_cats = SUB_CATEGORY_BY_CATEGORY[category]
        for i in range(count):
            brand = brands[i % len(brands)]
            sub_category = sub_cats[i % len(sub_cats)]
            pack_size = random.choice(PACK_SIZES_ML)

            # Unit price scales loosely with pack size and category premium-ness
            base_price_per_100ml = {
                "Juice": 6.5,
                "Water": 2.0,
                "Carbonated": 4.5,
                "Energy Drink": 12.0,
                "Dairy": 7.0,
            }[category]
            unit_price = round((pack_size / 100) * base_price_per_100ml *
                                np.random.uniform(0.9, 1.15), 2)

            # First two SKUs defined per category are "hero" products (high
            # demand); remaining alternate medium/low so every category has
            # a realistic long tail.
            if i == 0:
                demand_tier = "high"
            elif i == 1:
                demand_tier = "medium"
            elif i % 2 == 0:
                demand_tier = "medium"
            else:
                demand_tier = "low"

            product_name = f"{brand} {sub_category} {pack_size}ml"

            products.append({
                "product_id": f"P{product_id_counter:03d}",
                "product_name": product_name,
                "brand": brand,
                "category": category,
                "sub_category": sub_category,
                "pack_size_ml": pack_size,
                "unit_price": unit_price,
                "_demand_tier": demand_tier,  # internal use only
            })
            product_id_counter += 1

    return pd.DataFrame(products)


product_df = build_product_master()

# ---------------------------------------------------------------------------
# 2. STORE MASTER
# ---------------------------------------------------------------------------
# 30 stores across 4 regions. Region count is uneven (reflects population /
# distribution density) and each region gets a "category affinity" used to
# bias sales volumes realistically (e.g. South prefers Carbonated + Juice,
# North leans Water + Dairy, etc.)

REGIONS = ["North", "South", "East", "West"]

REGION_CITIES = {
    "North": ["Delhi", "Lucknow", "Chandigarh", "Jaipur"],
    "South": ["Bengaluru", "Chennai", "Hyderabad", "Kochi"],
    "East": ["Kolkata", "Patna", "Bhubaneswar", "Guwahati"],
    "West": ["Mumbai", "Pune", "Ahmedabad", "Surat"],
}

STORE_FORMATS = ["Supermarket", "Convenience Store", "Hypermarket", "Kirana/Local Mart"]

# Region demand multipliers (overall market size per region) and category
# affinity multipliers (which categories over/under-index in that region).
REGION_BASE_MULTIPLIER = {
    "North": 1.05,
    "South": 1.15,   # largest market in this synthetic dataset
    "East": 0.85,
    "West": 1.00,
}

REGION_CATEGORY_AFFINITY = {
    "North": {"Water": 1.25, "Dairy": 1.20, "Juice": 0.95, "Carbonated": 0.90, "Energy Drink": 0.90},
    "South": {"Carbonated": 1.30, "Juice": 1.20, "Water": 1.00, "Dairy": 0.85, "Energy Drink": 1.05},
    "East":  {"Dairy": 1.15, "Juice": 1.05, "Water": 0.95, "Carbonated": 0.90, "Energy Drink": 0.80},
    "West":  {"Energy Drink": 1.35, "Carbonated": 1.10, "Water": 1.00, "Juice": 0.95, "Dairy": 0.90},
}

STORE_FORMAT_MULTIPLIER = {
    "Hypermarket": 1.6,
    "Supermarket": 1.2,
    "Convenience Store": 0.7,
    "Kirana/Local Mart": 0.5,
}


def build_store_master():
    stores = []
    store_id_counter = 1
    # Distribute 30 stores across 4 regions, roughly proportional to market size
    region_store_counts = {"North": 8, "South": 9, "East": 6, "West": 7}
    assert sum(region_store_counts.values()) == 30

    for region, count in region_store_counts.items():
        cities = REGION_CITIES[region]
        for i in range(count):
            city = cities[i % len(cities)]
            store_format = random.choices(
                STORE_FORMATS,
                weights=[0.30, 0.25, 0.20, 0.25],
                k=1
            )[0]
            store_name = f"{city} {store_format} #{i + 1}"
            stores.append({
                "store_id": f"S{store_id_counter:03d}",
                "store_name": store_name,
                "region": region,
                "city": city,
                "store_format": store_format,
            })
            store_id_counter += 1

    return pd.DataFrame(stores)


store_df = build_store_master()

# ---------------------------------------------------------------------------
# 3. SALES & PROMOTIONS  +  4. INVENTORY  (generated together, week by week)
# ---------------------------------------------------------------------------
# Sales and inventory are generated jointly per (week, product, store) because
# inventory depends directly on that week's units_sold, and stockouts depend
# on whether opening stock + receipts could cover demand. This is what makes
# the "stockout analysis" use case meaningful: a stockout is not random, it's
# what happens when a demand spike (often promo-driven) exceeds available
# stock.

DEMAND_TIER_BASE_UNITS = {
    "high": 140,
    "medium": 70,
    "low": 25,
}

PROMOTION_TYPES = ["Price Cut", "Bundle", "BOGO", "Display Feature"]

# Promotion effect model: each promo type has a (volume_lift, discount_range)
# profile. BOGO drives the biggest volume lift but the heaviest discount
# (effectively ~50% off per unit economics), so revenue lift is much smaller
# than volume lift, and margin is worst. Display Feature is a cheap, modest
# lever - good for steady-sellers, weak for already-saturated hero SKUs.
PROMO_EFFECT = {
    "Price Cut":       {"volume_lift": (1.25, 1.45), "discount_pct": (10, 20)},
    "Bundle":          {"volume_lift": (1.35, 1.60), "discount_pct": (15, 25)},
    "BOGO":            {"volume_lift": (1.70, 2.10), "discount_pct": (45, 50)},
    "Display Feature": {"volume_lift": (1.10, 1.25), "discount_pct": (0, 5)},
}

# Weekly seasonality index (week 1..24 spans roughly Jan-Jun in this dataset).
# Beverages (esp. Water, Carbonated, Energy Drink) trend up sharply as the
# weather warms heading into summer; Juice and Dairy are comparatively flat.
def seasonality_index(week_num, category):
    """Returns a multiplier >0 representing seasonal demand strength."""
    # Base seasonal curve: warms up steadily from week 1 to week 24
    # (Jan -> late Jun), simulating rising temperature/demand.
    progress = week_num / NUM_WEEKS  # 0 -> ~1
    warm_curve = 0.85 + 0.45 * progress  # 0.85 at week1 -> ~1.30 at week24

    category_sensitivity = {
        "Water": 1.3,
        "Energy Drink": 1.15,
        "Carbonated": 1.1,
        "Juice": 0.6,
        "Dairy": 0.3,
    }[category]

    # Scale the deviation from 1.0 by category sensitivity so heat-sensitive
    # categories swing more, flat categories swing less.
    deviation = (warm_curve - 1.0) * category_sensitivity
    return 1.0 + deviation


def generate_sales_and_inventory(product_df, store_df):
    sales_rows = []
    inventory_rows = []

    # Track running closing stock per (product, store) across weeks so that
    # this week's opening_stock = last week's closing_stock.
    running_stock = {}

    # Initialize opening stock generously (2-3 weeks of expected high-tier
    # demand) so week 1 doesn't artificially stock out everywhere.
    for _, prod in product_df.iterrows():
        base_units = DEMAND_TIER_BASE_UNITS[prod["_demand_tier"]]
        for _, store in store_df.iterrows():
            fmt_mult = STORE_FORMAT_MULTIPLIER[store["store_format"]]
            init_stock = int(base_units * fmt_mult * np.random.uniform(1.8, 2.6))
            running_stock[(prod["product_id"], store["store_id"])] = init_stock

    for week_num in range(1, NUM_WEEKS + 1):
        week_start_date = (START_DATE + timedelta(weeks=week_num - 1)).strftime("%Y-%m-%d")

        # Decide which (product, store) combinations run a promotion this
        # week. Roughly 18% of product-store-weeks carry a promotion -
        # promotions are usually run on a subset of stores/products at a
        # time, not everywhere at once.
        for _, prod in product_df.iterrows():
            product_id = prod["product_id"]
            category = prod["category"]
            unit_price = prod["unit_price"]
            demand_tier = prod["_demand_tier"]
            base_units = DEMAND_TIER_BASE_UNITS[demand_tier]
            season_mult = seasonality_index(week_num, category)

            for _, store in store_df.iterrows():
                store_id = store["store_id"]
                region = store["region"]
                fmt_mult = STORE_FORMAT_MULTIPLIER[store["store_format"]]
                region_mult = REGION_BASE_MULTIPLIER[region]
                affinity_mult = REGION_CATEGORY_AFFINITY[region][category]

                # ---- Decide promotion status ----
                is_promo = np.random.random() < 0.18
                promotion_type = ""
                discount_pct = 0.0
                promo_volume_lift = 1.0

                if is_promo:
                    promotion_type = random.choice(PROMOTION_TYPES)
                    effect = PROMO_EFFECT[promotion_type]
                    promo_volume_lift = np.random.uniform(*effect["volume_lift"])
                    discount_pct = round(np.random.uniform(*effect["discount_pct"]), 1)

                # ---- Base demand calculation ----
                # Combine: SKU tier base * format size * region market size *
                # regional category affinity * seasonality * promo lift *
                # random week-to-week noise.
                noise = np.random.normal(loc=1.0, scale=0.18)
                noise = max(noise, 0.4)  # floor so noise never goes negative/absurd

                expected_demand = (
                    base_units * fmt_mult * region_mult * affinity_mult *
                    season_mult * promo_volume_lift * noise
                )

                # Add slow-mover dampening: low-tier SKUs occasionally see
                # near-zero demand in smaller formats (realistic long tail).
                if demand_tier == "low" and store["store_format"] in (
                    "Kirana/Local Mart", "Convenience Store"
                ):
                    expected_demand *= np.random.uniform(0.5, 0.85)

                desired_units = max(int(round(expected_demand)), 0)

                # ---- Inventory mechanics ----
                opening_stock = running_stock[(product_id, store_id)]

                # Replenishment: stores typically reorder to cover ~1.4x of
                # the PRIOR week's actual sell-through, with lead-time noise.
                # This intentionally creates situations where a sudden promo
                # spike outruns the replenishment that was planned before the
                # promo was confirmed - the realistic root cause of stockouts.
                planned_receipt = int(round(base_units * fmt_mult * region_mult *
                                             affinity_mult * np.random.uniform(0.95, 1.35)))
                # Occasionally (10% of the time) a supply hiccup cuts receipts hard
                # (e.g. distributor delay, production shortfall).
                if np.random.random() < 0.10:
                    planned_receipt = int(planned_receipt * np.random.uniform(0.3, 0.6))
                units_received = max(planned_receipt, 0)

                available_units = opening_stock + units_received

                # Actual units sold is capped by what's available -> this is
                # exactly how a stockout emerges: desired demand > available.
                units_sold = min(desired_units, available_units)
                stockout_flag = 1 if desired_units > available_units else 0

                closing_stock = available_units - units_sold
                running_stock[(product_id, store_id)] = closing_stock

                revenue = round(units_sold * unit_price * (1 - discount_pct / 100), 2)

                sales_rows.append({
                    "week_start_date": week_start_date,
                    "product_id": product_id,
                    "store_id": store_id,
                    "region": region,
                    "units_sold": units_sold,
                    "revenue": revenue,
                    "promotion_flag": int(is_promo),
                    "promotion_type": promotion_type,
                    "discount_pct": discount_pct,
                })

                inventory_rows.append({
                    "week_start_date": week_start_date,
                    "product_id": product_id,
                    "store_id": store_id,
                    "opening_stock": opening_stock,
                    "units_received": units_received,
                    "units_sold": units_sold,
                    "closing_stock": closing_stock,
                    "stockout_flag": stockout_flag,
                })

    return pd.DataFrame(sales_rows), pd.DataFrame(inventory_rows)


sales_df, inventory_df = generate_sales_and_inventory(product_df, store_df)

# Drop internal-only helper column before export
product_export_df = product_df.drop(columns=["_demand_tier"])

# ---------------------------------------------------------------------------
# 5. VALIDATION CHECKS
# ---------------------------------------------------------------------------
# Basic data-quality gate. If any check fails, we raise loudly rather than
# silently shipping a broken dataset - this mirrors what a real data
# engineering pipeline should do before publishing a dataset to consumers.

def run_validation_checks(product_df, store_df, sales_df, inventory_df):
    errors = []

    # --- Row counts ---
    if len(product_df) != 20:
        errors.append(f"Expected 20 products, got {len(product_df)}")
    if len(store_df) != 30:
        errors.append(f"Expected 30 stores, got {len(store_df)}")

    expected_fact_rows = 20 * 30 * NUM_WEEKS
    if len(sales_df) != expected_fact_rows:
        errors.append(f"Expected {expected_fact_rows} sales rows, got {len(sales_df)}")
    if len(inventory_df) != expected_fact_rows:
        errors.append(f"Expected {expected_fact_rows} inventory rows, got {len(inventory_df)}")

    # --- Primary key uniqueness ---
    if product_df["product_id"].duplicated().any():
        errors.append("Duplicate product_id found in product_master")
    if store_df["store_id"].duplicated().any():
        errors.append("Duplicate store_id found in store_master")

    # --- Referential integrity: every product_id/store_id in fact tables
    #     must exist in the corresponding master table ---
    valid_products = set(product_df["product_id"])
    valid_stores = set(store_df["store_id"])

    if not set(sales_df["product_id"]).issubset(valid_products):
        errors.append("sales_promotions has product_id values not in product_master")
    if not set(sales_df["store_id"]).issubset(valid_stores):
        errors.append("sales_promotions has store_id values not in store_master")
    if not set(inventory_df["product_id"]).issubset(valid_products):
        errors.append("inventory has product_id values not in product_master")
    if not set(inventory_df["store_id"]).issubset(valid_stores):
        errors.append("inventory has store_id values not in store_master")

    # --- No negative values where they shouldn't exist ---
    if (sales_df["units_sold"] < 0).any():
        errors.append("Negative units_sold found in sales_promotions")
    if (sales_df["revenue"] < 0).any():
        errors.append("Negative revenue found in sales_promotions")
    if (inventory_df["opening_stock"] < 0).any():
        errors.append("Negative opening_stock found in inventory")
    if (inventory_df["closing_stock"] < 0).any():
        errors.append("Negative closing_stock found in inventory")

    # --- Inventory equation must balance exactly:
    #     closing_stock = opening_stock + units_received - units_sold ---
    recomputed_closing = (
        inventory_df["opening_stock"] + inventory_df["units_received"] - inventory_df["units_sold"]
    )
    if not (recomputed_closing == inventory_df["closing_stock"]).all():
        errors.append("Inventory equation does not balance for some rows "
                       "(closing_stock != opening_stock + units_received - units_sold)")

    # --- units_sold must match between sales_promotions and inventory for
    #     the same (week, product, store) ---
    merged_check = sales_df.merge(
        inventory_df,
        on=["week_start_date", "product_id", "store_id"],
        suffixes=("_sales", "_inv"),
    )
    if not (merged_check["units_sold_sales"] == merged_check["units_sold_inv"]).all():
        errors.append("units_sold mismatch between sales_promotions and inventory")

    # --- stockout_flag must be internally consistent: if stockout_flag = 1,
    #     closing_stock should be 0 (all available stock was sold) ---
    stockout_rows = inventory_df[inventory_df["stockout_flag"] == 1]
    if not (stockout_rows["closing_stock"] == 0).all():
        errors.append("Found stockout_flag=1 rows where closing_stock is not 0")

    # --- promotion_flag/promotion_type/discount_pct consistency ---
    promo_rows = sales_df[sales_df["promotion_flag"] == 1]
    non_promo_rows = sales_df[sales_df["promotion_flag"] == 0]
    if (promo_rows["promotion_type"] == "").any():
        errors.append("Found promotion_flag=1 rows with blank promotion_type")
    if (non_promo_rows["promotion_type"] != "").any():
        errors.append("Found promotion_flag=0 rows with a non-blank promotion_type")
    if (non_promo_rows["discount_pct"] != 0).any():
        errors.append("Found promotion_flag=0 rows with non-zero discount_pct")

    # --- date range sanity ---
    unique_weeks = sorted(sales_df["week_start_date"].unique())
    if len(unique_weeks) != NUM_WEEKS:
        errors.append(f"Expected {NUM_WEEKS} unique weeks, got {len(unique_weeks)}")

    if errors:
        raise AssertionError(
            "Dataset validation FAILED with the following issues:\n  - " +
            "\n  - ".join(errors)
        )

    print("All validation checks passed:")
    print(f"   - {len(product_df)} products, {len(store_df)} stores")
    print(f"   - {len(sales_df):,} sales_promotions rows across {len(unique_weeks)} weeks")
    print(f"   - {len(inventory_df):,} inventory rows")
    print(f"   - Referential integrity OK, inventory equation balances, "
          f"stockout logic consistent")


run_validation_checks(product_export_df, store_df, sales_df, inventory_df)

# ---------------------------------------------------------------------------
# 6. EXPORT
# ---------------------------------------------------------------------------
product_export_df.to_csv(os.path.join(OUTPUT_DIR, "product_master.csv"), index=False)
store_df.to_csv(os.path.join(OUTPUT_DIR, "store_master.csv"), index=False)
sales_df.to_csv(os.path.join(OUTPUT_DIR, "sales_promotions.csv"), index=False)
inventory_df.to_csv(os.path.join(OUTPUT_DIR, "inventory.csv"), index=False)

print("\nFiles written to ./data/:")
for fname in ["product_master.csv", "store_master.csv", "sales_promotions.csv", "inventory.csv"]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"   - {fname}  ({size_kb:.1f} KB)")

# ---------------------------------------------------------------------------
# 7. QUICK SUMMARY (sanity-check printout, not required for downstream use)
# ---------------------------------------------------------------------------
print("\nQuick sanity summary:")
print(f"   Total revenue across dataset: {sales_df['revenue'].sum():,.2f}")
print(f"   Overall stockout rate: {inventory_df['stockout_flag'].mean() * 100:.2f}%")
print(f"   Promotion incidence rate: {sales_df['promotion_flag'].mean() * 100:.2f}%")
top_products = (
    sales_df.merge(product_export_df, on="product_id")
    .groupby("product_name")["revenue"].sum()
    .sort_values(ascending=False)
    .head(3)
)
print("   Top 3 products by revenue:")
for name, rev in top_products.items():
    print(f"      - {name}: {rev:,.2f}")
