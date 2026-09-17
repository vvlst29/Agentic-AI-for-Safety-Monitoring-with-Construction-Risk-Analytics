# -*- coding: utf-8 -*-
"""insurance_policy_db_builder.py

Creates and populates insurance_policy.db — the live policy database
that DynamicInsuranceAgent._fetch_policy() reads from.

Mirrors the pattern used by resource_monitoring_agent.py: a
build_database() function that (re)creates tables and seeds them,
so it can be run standalone or imported and called by another agent
/ admin script whenever policy data changes.

Table: policies
    Policy_Number   TEXT   PRIMARY KEY   e.g. "CPM-2026-A-014"
    Category        TEXT                 e.g. "Equipment", "PPE/Safety", "Quality"
    Zone            TEXT                 e.g. "Zone A", "Zone B" ("ALL" = applies site-wide)
    Policy_Name     TEXT                 e.g. "Contractor's Plant & Machinery (CPM) Policy"
    Insurer         TEXT                 e.g. "ICICI Lombard"
    Coverage_Limit  REAL                 max payable amount (INR)
    Deductible      REAL                 amount borne before payout (INR)
    Effective_Date  TEXT                 "YYYY-MM-DD"
    Expiry_Date     TEXT                 "YYYY-MM-DD"
    Broker_Contact  TEXT                 name / phone / email of broker

DynamicInsuranceAgent looks up rows by (Category, Zone). If you want a
policy to apply to every zone, set Zone = "ALL" and update the agent's
query to also check for "ALL" (see note at bottom of this file).
"""

import sqlite3
import os

DB_PATH = "insurance_policy.db"


def build_database(db_path: str = DB_PATH, reset: bool = True):
    """Creates the policies table and seeds it with sample data.

    Set reset=False to keep existing data and only create the table
    if missing (useful once you're managing real records)."""

    if reset and os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            Policy_Number   TEXT PRIMARY KEY,
            Category        TEXT NOT NULL,
            Zone            TEXT NOT NULL,
            Policy_Name     TEXT NOT NULL,
            Insurer         TEXT NOT NULL,
            Coverage_Limit  REAL NOT NULL,
            Deductible      REAL NOT NULL,
            Effective_Date  TEXT,
            Expiry_Date     TEXT,
            Broker_Contact  TEXT
        )
    """)

    # ----------------------------------------------------------------
    # Sample seed data — replace with your real policy schedule.
    # ----------------------------------------------------------------
    sample_policies = [
        ("WC-2026-A-001", "PPE/Safety", "Zone A", "Workmen's Compensation Policy",
         "ICICI Lombard", 2_000_000.0, 5_000.0, "2026-01-01", "2026-12-31", "Ravi Shah - broker@icicilombard.com"),
        ("WC-2026-B-002", "PPE/Safety", "Zone B", "Workmen's Compensation Policy",
         "ICICI Lombard", 2_000_000.0, 5_000.0, "2026-01-01", "2026-12-31", "Ravi Shah - broker@icicilombard.com"),
        ("CPM-2026-A-014", "Equipment", "Zone A", "Contractor's Plant & Machinery (CPM) Policy",
         "Tata AIG", 5_000_000.0, 25_000.0, "2026-01-01", "2026-12-31", "Neha Verma - neha@tataaig.com"),
        ("CPM-2026-C-015", "Equipment", "Zone C", "Contractor's Plant & Machinery (CPM) Policy",
         "Tata AIG", 4_000_000.0, 20_000.0, "2026-01-01", "2026-12-31", "Neha Verma - neha@tataaig.com"),
        ("CAR-2026-ALL-003", "Quality", "ALL", "Contractor's All Risk (CAR) Policy",
         "HDFC Ergo", 10_000_000.0, 50_000.0, "2026-01-01", "2026-12-31", "Amit Joshi - amit@hdfcergo.com"),
        ("GL-2026-ALL-009", "General", "ALL", "General Liability Policy",
         "Bajaj Allianz", 1_000_000.0, 10_000.0, "2026-01-01", "2026-12-31", "Priya Nair - priya@bajajallianz.com"),
    ]

    cur.executemany("""
        INSERT OR REPLACE INTO policies
        (Policy_Number, Category, Zone, Policy_Name, Insurer, Coverage_Limit,
         Deductible, Effective_Date, Expiry_Date, Broker_Contact)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_policies)

    conn.commit()
    conn.close()
    print(f"[insurance_policy_db_builder] Built '{db_path}' with {len(sample_policies)} policy record(s).")


if __name__ == "__main__":
    build_database()
