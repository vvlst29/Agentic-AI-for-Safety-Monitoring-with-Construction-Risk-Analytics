import argparse
import json
import os
import random
import sqlite3
import sys
import time
from datetime import datetime

import pandas as pd

DB_PATH = "resource_simulation.db"
REPORTS_DIR = "reports"

LABOUR_FILE = "Labour.xlsx"
EQUIPMENT_FILE = "Equipment.xlsx"
MATERIAL_FILE = "Material.xlsx"


# --------------------------------------------------------------------------
# Database setup
# --------------------------------------------------------------------------
def build_database(db_path=DB_PATH, force_reload=False):
    """Loads the three Excel files into SQLite tables (acts as the 'live DB').
    If the DB already exists and force_reload is False, it is reused as-is
    so that simulated changes persist across restarts."""
    fresh = force_reload or not os.path.exists(db_path)
    conn = sqlite3.connect(db_path)

    if fresh:
        labour = pd.read_excel(LABOUR_FILE)
        equipment = pd.read_excel(EQUIPMENT_FILE)
        material = pd.read_excel(MATERIAL_FILE)

        labour.to_sql("labour", conn, if_exists="replace", index=False)
        equipment.to_sql("equipment", conn, if_exists="replace", index=False)
        material.to_sql("material", conn, if_exists="replace", index=False)
        conn.commit()
        print(f"[DB] Loaded fresh data into {db_path}")
    else:
        print(f"[DB] Reusing existing simulated database {db_path}")

    return conn


# --------------------------------------------------------------------------
# Simulation: mutate the "live" database each cycle
# --------------------------------------------------------------------------
def simulate_labour_changes(conn):
    df = pd.read_sql("SELECT rowid AS rid, * FROM labour", conn)
    for _, row in df.iterrows():
        delta = random.randint(-3, 3)  # workers arriving / leaving site
        new_val = max(0, row["Available_Labour"] + delta)
        conn.execute(
            "UPDATE labour SET Available_Labour = ? WHERE rowid = ?",
            (int(new_val), int(row["rid"])),
        )
    conn.commit()


def simulate_equipment_changes(conn):
    df = pd.read_sql("SELECT rowid AS rid, * FROM equipment", conn)
    for _, row in df.iterrows():
        delta = random.randint(-2, 2)  # units arriving / leaving / breaking down
        new_available = max(0, row["Available_Quantity"] + delta)

        # Occasionally move a unit into or out of maintenance
        maint_delta = random.choice([-1, 0, 0, 0, 1])
        new_maint = min(new_available, max(0, row["Under_Maintenance"] + maint_delta))
        new_working = max(0, new_available - new_maint)

        conn.execute(
            """UPDATE equipment
               SET Available_Quantity = ?, Under_Maintenance = ?, Working_Quantity = ?
               WHERE rowid = ?""",
            (int(new_available), int(new_maint), int(new_working), int(row["rid"])),
        )
    conn.commit()


def simulate_material_changes(conn):
    df = pd.read_sql("SELECT rowid AS rid, * FROM material", conn)
    for _, row in df.iterrows():
        # Materials trend downward (consumption) with occasional restock spikes
        if random.random() < 0.15:
            pct_delta = random.uniform(0.10, 0.40)  # restock delivery
        else:
            pct_delta = random.uniform(-0.12, 0.02)  # normal consumption
        new_val = max(0, int(row["Available_Quantity"] * (1 + pct_delta)))
        conn.execute(
            "UPDATE material SET Available_Quantity = ? WHERE rowid = ?",
            (int(new_val), int(row["rid"])),
        )
    conn.commit()


# --------------------------------------------------------------------------
# Status / risk classification helpers
# --------------------------------------------------------------------------
def classify_status(available, required):
    if available <= 0:
        return "Unavailable"
    if available >= required:
        return "Available"
    return "Partially Available"


def classify_equipment_risk(availability_pct, maintenance_ratio):
    """Simple, transparent risk model:
    High   -> availability under 50% OR over 40% of the fleet is under maintenance
    Medium -> availability under 80% OR over 20% under maintenance
    Low    -> otherwise"""
    if availability_pct < 50 or maintenance_ratio > 0.40:
        return "High"
    if availability_pct < 80 or maintenance_ratio > 0.20:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------
# Metric computation
# --------------------------------------------------------------------------
def compute_labour_metrics(conn):
    df = pd.read_sql("SELECT * FROM labour", conn)

    total_required = int(df["Required_Labour"].sum())
    total_available = int(df["Available_Labour"].sum())
    availability_pct = round(100 * total_available / total_required, 1) if total_required else 0.0
    shortage = max(0, total_required - total_available)

    zone = (
        df.groupby("Zone")
        .agg(Required_Labour=("Required_Labour", "sum"), Available_Labour=("Available_Labour", "sum"))
        .reset_index()
    )
    zone["Availability_%"] = (100 * zone["Available_Labour"] / zone["Required_Labour"]).round(1)
    zone["Shortage"] = (zone["Required_Labour"] - zone["Available_Labour"]).clip(lower=0)
    zone["Zone_Status"] = zone.apply(
        lambda r: classify_status(r["Available_Labour"], r["Required_Labour"]), axis=1
    )

    return {
        "total_required": total_required,
        "total_available": total_available,
        "availability_pct": availability_pct,
        "shortage": int(shortage),
        "zone_availability": zone.to_dict(orient="records"),
    }


def compute_equipment_metrics(conn):
    df = pd.read_sql("SELECT * FROM equipment", conn)

    total_required = int(df["Required_Quantity"].sum())
    total_available = int(df["Available_Quantity"].sum())
    total_maintenance = int(df["Under_Maintenance"].sum())
    availability_pct = round(100 * total_available / total_required, 1) if total_required else 0.0
    shortage = max(0, total_required - total_available)
    maintenance_ratio = total_maintenance / total_available if total_available else 0.0
    overall_risk = classify_equipment_risk(availability_pct, maintenance_ratio)

    zone = (
        df.groupby("Zone")
        .agg(
            Required_Quantity=("Required_Quantity", "sum"),
            Available_Quantity=("Available_Quantity", "sum"),
            Under_Maintenance=("Under_Maintenance", "sum"),
        )
        .reset_index()
    )
    zone["Availability_%"] = (100 * zone["Available_Quantity"] / zone["Required_Quantity"]).round(1)
    zone["Shortage"] = (zone["Required_Quantity"] - zone["Available_Quantity"]).clip(lower=0)
    zone["Maintenance_Ratio"] = (
        zone["Under_Maintenance"] / zone["Available_Quantity"].replace(0, pd.NA)
    ).fillna(0)
    zone["Risk"] = zone.apply(
        lambda r: classify_equipment_risk(r["Availability_%"], r["Maintenance_Ratio"]), axis=1
    )
    zone = zone.drop(columns=["Maintenance_Ratio"])

    return {
        "total_required": total_required,
        "total_available": total_available,
        "availability_pct": availability_pct,
        "shortage": int(shortage),
        "overall_risk": overall_risk,
        "zone_availability": zone.to_dict(orient="records"),
    }


def compute_material_metrics(conn):
    df = pd.read_sql("SELECT * FROM material", conn)

    total_required = int(df["Required_Quantity"].sum())
    total_available = int(df["Available_Quantity"].sum())
    availability_pct = round(100 * total_available / total_required, 1) if total_required else 0.0
    shortage = max(0, total_required - total_available)

    zone = (
        df.groupby("Zone")
        .agg(Required_Quantity=("Required_Quantity", "sum"), Available_Quantity=("Available_Quantity", "sum"))
        .reset_index()
    )
    zone["Availability_%"] = (100 * zone["Available_Quantity"] / zone["Required_Quantity"]).round(1)
    zone["Shortage"] = (zone["Required_Quantity"] - zone["Available_Quantity"]).clip(lower=0)
    zone["Zone_Status"] = zone.apply(
        lambda r: classify_status(r["Available_Quantity"], r["Required_Quantity"]), axis=1
    )

    return {
        "total_required": total_required,
        "total_available": total_available,
        "availability_pct": availability_pct,
        "shortage": int(shortage),
        "zone_availability": zone.to_dict(orient="records"),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def print_dashboard(timestamp, labour, equipment, material):
    line = "=" * 72
    print(f"\n{line}\nRESOURCE DASHBOARD — refreshed {timestamp}\n{line}")

    print("\n[LABOUR]")
    print(f"  Available: {labour['total_available']} / Required: {labour['total_required']}")
    print(f"  Availability %: {labour['availability_pct']}%   Shortage: {labour['shortage']}")
    print("  Zone-wise:")
    for z in labour["zone_availability"]:
        print(
            f"    {z['Zone']:<8} Available={z['Available_Labour']:<5} "
            f"Required={z['Required_Labour']:<5} "
            f"Avail%={z['Availability_%']:<6} Shortage={z['Shortage']:<5} Status={z['Zone_Status']}"
        )

    print("\n[EQUIPMENT]")
    print(f"  Available: {equipment['total_available']} / Required: {equipment['total_required']}")
    print(f"  Availability %: {equipment['availability_pct']}%   Shortage: {equipment['shortage']}")
    print(f"  Overall Risk: {equipment['overall_risk']}")
    print("  Zone-wise:")
    for z in equipment["zone_availability"]:
        print(
            f"    {z['Zone']:<8} Available={z['Available_Quantity']:<5} "
            f"Required={z['Required_Quantity']:<5} "
            f"Avail%={z['Availability_%']:<6} Shortage={z['Shortage']:<5} "
            f"UnderMaint={z['Under_Maintenance']:<4} Risk={z['Risk']}"
        )

    print("\n[MATERIAL]")
    print(f"  Available: {material['total_available']} / Required: {material['total_required']}")
    print(f"  Availability %: {material['availability_pct']}%   Shortage: {material['shortage']}")
    print("  Zone-wise:")
    for z in material["zone_availability"]:
        print(
            f"    {z['Zone']:<8} Available={z['Available_Quantity']:<5} "
            f"Required={z['Required_Quantity']:<5} "
            f"Avail%={z['Availability_%']:<6} Shortage={z['Shortage']:<5} Status={z['Zone_Status']}"
        )
    print(line)


def save_snapshot(timestamp, labour, equipment, material):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    safe_ts = timestamp.replace(":", "-").replace(" ", "_")

    payload = {"timestamp": timestamp, "labour": labour, "equipment": equipment, "material": material}
    json_path = os.path.join(REPORTS_DIR, f"snapshot_{safe_ts}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Also drop a flat CSV summary (one row per zone per resource type) for easy charting
    rows = []
    for z in labour["zone_availability"]:
        rows.append({"Resource": "Labour", "Zone": z["Zone"], "Available": z["Available_Labour"],
                      "Required": z["Required_Labour"], "Availability_%": z["Availability_%"],
                      "Shortage": z["Shortage"], "Status_or_Risk": z["Zone_Status"]})
    for z in equipment["zone_availability"]:
        rows.append({"Resource": "Equipment", "Zone": z["Zone"], "Available": z["Available_Quantity"],
                      "Required": z["Required_Quantity"], "Availability_%": z["Availability_%"],
                      "Shortage": z["Shortage"], "Status_or_Risk": z["Risk"]})
    for z in material["zone_availability"]:
        rows.append({"Resource": "Material", "Zone": z["Zone"], "Available": z["Available_Quantity"],
                      "Required": z["Required_Quantity"], "Availability_%": z["Availability_%"],
                      "Shortage": z["Shortage"], "Status_or_Risk": z["Zone_Status"]})

    csv_path = os.path.join(REPORTS_DIR, f"snapshot_{safe_ts}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"[Saved] {json_path}\n[Saved] {csv_path}")


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def run_cycle(conn):
    simulate_labour_changes(conn)
    simulate_equipment_changes(conn)
    simulate_material_changes(conn)

    labour = compute_labour_metrics(conn)
    equipment = compute_equipment_metrics(conn)
    material = compute_material_metrics(conn)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_dashboard(timestamp, labour, equipment, material)
    save_snapshot(timestamp, labour, equipment, material)


def main():
    parser = argparse.ArgumentParser(description="Resource Monitoring Agent")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between refreshes (default 300 = 5 min)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--reload", action="store_true", help="Reload the Excel files into a fresh database")
    args = parser.parse_args()

    conn = build_database(force_reload=args.reload)

    if args.once:
        run_cycle(conn)
        conn.close()
        return

    print(f"Agent started. Refreshing every {args.interval} seconds. Press Ctrl+C to stop.")
    try:
        while True:
            run_cycle(conn)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
