-- ============================================================================
-- schema.sql
-- FMCG Beverages Business Insights Assistant - Database Schema
-- Engine: SQLite 3
-- ============================================================================
-- Design notes:
--   * product_master and store_master are dimension tables (slowly changing,
--     small row counts).
--   * sales_promotions and inventory are fact tables at WEEK x PRODUCT x STORE
--     grain. They share the same grain, which is what allows the AI assistant
--     to easily join sales with inventory for combined stockout/promo
--     analysis.
--   * Foreign keys enforce referential integrity back to the dimension
--     tables. SQLite does not enforce FKs by default, so the setup script
--     turns PRAGMA foreign_keys = ON explicitly.
--   * Indexes are added on the columns the assistant's generated SQL will
--     filter/group by most often: region, product_id, store_id,
--     week_start_date, promotion_type, stockout_flag.
-- ============================================================================

PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------------------------
-- 1. PRODUCT MASTER
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS product_master;

CREATE TABLE product_master (
    product_id      TEXT PRIMARY KEY,
    product_name    TEXT NOT NULL,
    brand           TEXT NOT NULL,
    category        TEXT NOT NULL CHECK (category IN
                        ('Juice', 'Water', 'Carbonated', 'Energy Drink', 'Dairy')),
    sub_category    TEXT NOT NULL,
    pack_size_ml    INTEGER NOT NULL CHECK (pack_size_ml > 0),
    unit_price      REAL NOT NULL CHECK (unit_price >= 0)
);

CREATE INDEX idx_product_category ON product_master (category);
CREATE INDEX idx_product_brand    ON product_master (brand);

-- ----------------------------------------------------------------------------
-- 2. STORE MASTER
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS store_master;

CREATE TABLE store_master (
    store_id        TEXT PRIMARY KEY,
    store_name      TEXT NOT NULL,
    region          TEXT NOT NULL CHECK (region IN ('North', 'South', 'East', 'West')),
    city            TEXT NOT NULL,
    store_format    TEXT NOT NULL CHECK (store_format IN
                        ('Supermarket', 'Convenience Store', 'Hypermarket', 'Kirana/Local Mart'))
);

CREATE INDEX idx_store_region ON store_master (region);
CREATE INDEX idx_store_format ON store_master (store_format);

-- ----------------------------------------------------------------------------
-- 3. SALES & PROMOTIONS (fact table, grain = week x product x store)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS sales_promotions;

CREATE TABLE sales_promotions (
    sales_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start_date     TEXT NOT NULL,         -- ISO format YYYY-MM-DD
    product_id          TEXT NOT NULL,
    store_id            TEXT NOT NULL,
    region              TEXT NOT NULL CHECK (region IN ('North', 'South', 'East', 'West')),
    units_sold          INTEGER NOT NULL CHECK (units_sold >= 0),
    revenue             REAL NOT NULL CHECK (revenue >= 0),
    promotion_flag      INTEGER NOT NULL CHECK (promotion_flag IN (0, 1)),
    promotion_type      TEXT CHECK (promotion_type IN
                            ('', 'Price Cut', 'Bundle', 'BOGO', 'Display Feature')),
    discount_pct        REAL NOT NULL DEFAULT 0 CHECK (discount_pct >= 0 AND discount_pct <= 100),

    FOREIGN KEY (product_id) REFERENCES product_master (product_id),
    FOREIGN KEY (store_id)   REFERENCES store_master (store_id),

    -- A given product/store/week combination should appear exactly once
    UNIQUE (week_start_date, product_id, store_id)
);

CREATE INDEX idx_sales_week        ON sales_promotions (week_start_date);
CREATE INDEX idx_sales_product     ON sales_promotions (product_id);
CREATE INDEX idx_sales_store       ON sales_promotions (store_id);
CREATE INDEX idx_sales_region      ON sales_promotions (region);
CREATE INDEX idx_sales_promo_flag  ON sales_promotions (promotion_flag);
CREATE INDEX idx_sales_promo_type  ON sales_promotions (promotion_type);
-- Composite index to accelerate the very common "trend over time for a
-- product/region" query pattern.
CREATE INDEX idx_sales_week_region_product ON sales_promotions (week_start_date, region, product_id);

-- ----------------------------------------------------------------------------
-- 4. INVENTORY (fact table, same grain as sales_promotions)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS inventory;

CREATE TABLE inventory (
    inventory_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start_date      TEXT NOT NULL,
    product_id            TEXT NOT NULL,
    store_id               TEXT NOT NULL,
    opening_stock        INTEGER NOT NULL CHECK (opening_stock >= 0),
    units_received       INTEGER NOT NULL CHECK (units_received >= 0),
    units_sold             INTEGER NOT NULL CHECK (units_sold >= 0),
    closing_stock         INTEGER NOT NULL CHECK (closing_stock >= 0),
    stockout_flag         INTEGER NOT NULL CHECK (stockout_flag IN (0, 1)),

    FOREIGN KEY (product_id) REFERENCES product_master (product_id),
    FOREIGN KEY (store_id)   REFERENCES store_master (store_id),

    UNIQUE (week_start_date, product_id, store_id)
);

CREATE INDEX idx_inventory_week       ON inventory (week_start_date);
CREATE INDEX idx_inventory_product    ON inventory (product_id);
CREATE INDEX idx_inventory_store      ON inventory (store_id);
CREATE INDEX idx_inventory_stockout   ON inventory (stockout_flag);
CREATE INDEX idx_inventory_week_product ON inventory (week_start_date, product_id);

-- ----------------------------------------------------------------------------
-- 5. CONVENIENCE VIEW (optional but useful)
-- ----------------------------------------------------------------------------
-- Pre-joined view combining sales + inventory + product + store dimensions.
-- This is what the LLM-generated SQL will often query against, since most
-- business questions need attributes from more than one table at once.
DROP VIEW IF EXISTS vw_sales_full;

CREATE VIEW vw_sales_full AS
SELECT
    sp.sales_id,
    sp.week_start_date,
    sp.product_id,
    pm.product_name,
    pm.brand,
    pm.category,
    pm.sub_category,
    pm.pack_size_ml,
    pm.unit_price,
    sp.store_id,
    sm.store_name,
    sp.region,
    sm.city,
    sm.store_format,
    sp.units_sold,
    sp.revenue,
    sp.promotion_flag,
    sp.promotion_type,
    sp.discount_pct,
    inv.opening_stock,
    inv.units_received,
    inv.closing_stock,
    inv.stockout_flag
FROM sales_promotions sp
JOIN product_master pm ON sp.product_id = pm.product_id
JOIN store_master sm   ON sp.store_id = sm.store_id
LEFT JOIN inventory inv
    ON sp.week_start_date = inv.week_start_date
   AND sp.product_id = inv.product_id
   AND sp.store_id = inv.store_id;
