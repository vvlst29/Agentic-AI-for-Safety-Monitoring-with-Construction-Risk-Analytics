import json

from schedule_agent import ScheduleAgent, get_resource_data, TASK_ZONE_MAP
from resource_monitoring_agent import DB_PATH, build_database

import pandas as pd

# --------------------------------------------------------------------------
# RATE CONSTANTS
# TODO: replace every value below with the real project/contract rates.
# These are placeholders only so the agent runs end-to-end - do NOT use
# them for real cost decisions.
# --------------------------------------------------------------------------
CURRENCY = "INR"

DAILY_SITE_OVERHEAD_RATE = 15000          # cost per day of overall delay
LIQUIDATED_DAMAGES_RATE = 25000           # per day beyond the contractual buffer
CONTRACTUAL_DELAY_BUFFER_DAYS = 0         # days of delay allowed before LD kicks in

OVERTIME_RATE_PER_WORKER_DAY = 1500       # cost per worker short (covered via OT/temp hire)
IDLE_LABOR_RATE_PER_WORKER_DAY = 800      # cost per worker surplus (paid but underused)

EQUIPMENT_RENTAL_RATE_PER_UNIT_DAY = 3000 # cost per unit of equipment shortfall

MATERIAL_DELAYED_PREMIUM_PCT = 0.15       # % of material value, if "delayed"
MATERIAL_PENDING_PREMIUM_PCT = 0.05       # % of material value, if "pending"
MATERIAL_UNIT_VALUE_FALLBACK = 500        # used only if material value isn't available

SUBCONTRACTOR_RISK_THRESHOLD = 0.6        # below this, flagged as a risk (not costed)


# --------------------------------------------------------------------------
# Raw data helpers
# --------------------------------------------------------------------------
def get_equipment_material_detail(task_id):
    """Pulls the numeric equipment gap and material value that
    resource_agent's get_resource_data() collapses into a bool / status
    string. schedule_agent doesn't expose these, so we query the same DB
    directly (read-only, force_reload=False - never mutates simulation state).
    """
    zone = TASK_ZONE_MAP.get(task_id)

    try:
        conn = build_database(DB_PATH, force_reload=False)
    except Exception as e:
        return {"error": str(e)}

    try:
        equipment_df = pd.read_sql("SELECT * FROM equipment", conn)
        material_df = pd.read_sql("SELECT * FROM material", conn)
    finally:
        conn.close()

    if zone is not None:
        equipment_df = equipment_df[equipment_df["Zone"] == zone]
        material_df = material_df[material_df["Zone"] == zone]

    equipment_required = int(equipment_df["Required_Quantity"].sum())
    equipment_on_hand = int(equipment_df["Available_Quantity"].sum())
    equipment_gap = max(0, equipment_required - equipment_on_hand)

    material_required = int(material_df["Required_Quantity"].sum())
    material_on_hand = int(material_df["Available_Quantity"].sum())
    material_gap_units = max(0, material_required - material_on_hand)

    return {
        "equipment_required": equipment_required,
        "equipment_on_hand": equipment_on_hand,
        "equipment_gap": equipment_gap,
        "material_required": material_required,
        "material_on_hand": material_on_hand,
        "material_gap_units": material_gap_units,
    }


# --------------------------------------------------------------------------
# Cost computation - one function per category, each returns
# {"amount": float, "basis": str} so the reasoning is always visible.
# --------------------------------------------------------------------------
def compute_delay_cost(schedule_result):
    total_delay_days = schedule_result["total_delay_days"]

    if total_delay_days <= 0:
        return {"amount": 0.0, "basis": "no schedule delay"}

    overhead = total_delay_days * DAILY_SITE_OVERHEAD_RATE
    ld_days = max(0.0, total_delay_days - CONTRACTUAL_DELAY_BUFFER_DAYS)
    liquidated_damages = ld_days * LIQUIDATED_DAMAGES_RATE

    amount = overhead + liquidated_damages
    basis = (
        f"{total_delay_days} delay day(s) x overhead ({DAILY_SITE_OVERHEAD_RATE}/day)"
    )
    if ld_days > 0:
        basis += f" + {ld_days} day(s) beyond buffer x LD rate ({LIQUIDATED_DAMAGES_RATE}/day)"

    return {"amount": round(amount, 2), "basis": basis}


def compute_labor_cost(resource):
    assigned = resource.get("labor_assigned_count")
    required = resource.get("labor_required_count")

    if assigned is None or required is None:
        return {"amount": None, "basis": "labor data unavailable"}

    if assigned < required:
        gap = required - assigned
        amount = gap * OVERTIME_RATE_PER_WORKER_DAY
        basis = f"{gap} worker(s) short x overtime/temp-hire rate ({OVERTIME_RATE_PER_WORKER_DAY}/day)"
    elif assigned > required:
        surplus = assigned - required
        amount = surplus * IDLE_LABOR_RATE_PER_WORKER_DAY
        basis = f"{surplus} surplus worker(s) x idle labor rate ({IDLE_LABOR_RATE_PER_WORKER_DAY}/day)"
    else:
        amount, basis = 0.0, "labor exactly matched to requirement"

    return {"amount": round(amount, 2), "basis": basis}


def compute_equipment_cost(equipment_detail):
    if "error" in equipment_detail:
        return {"amount": None, "basis": f"equipment data unavailable ({equipment_detail['error']})"}

    gap = equipment_detail["equipment_gap"]
    if gap <= 0:
        return {"amount": 0.0, "basis": "no equipment shortfall"}

    amount = gap * EQUIPMENT_RENTAL_RATE_PER_UNIT_DAY
    basis = f"{gap} unit(s) short x rental rate ({EQUIPMENT_RENTAL_RATE_PER_UNIT_DAY}/unit-day)"
    return {"amount": round(amount, 2), "basis": basis}


def compute_material_cost(resource, equipment_detail):
    status = resource.get("material_delivery_status")
    if status is None:
        return {"amount": None, "basis": "material data unavailable"}

    if "error" in equipment_detail:
        gap_units = None
    else:
        gap_units = equipment_detail.get("material_gap_units")

    if status == "on_time":
        return {"amount": 0.0, "basis": "material delivery on time"}

    premium_pct = MATERIAL_DELAYED_PREMIUM_PCT if status == "delayed" else MATERIAL_PENDING_PREMIUM_PCT

    if gap_units:
        exposed_value = gap_units * MATERIAL_UNIT_VALUE_FALLBACK
    else:
        exposed_value = MATERIAL_UNIT_VALUE_FALLBACK  # fallback if gap unknown

    amount = exposed_value * premium_pct
    basis = (
        f"material '{status}' - {premium_pct * 100:.0f}% expedite premium on "
        f"~{exposed_value} exposed value (unit value is a placeholder, replace with real pricing)"
    )
    return {"amount": round(amount, 2), "basis": basis}


def compute_subcontractor_risk(resource):
    score = resource.get("subcontractor_reliability_score")
    if score is None:
        return {"score": None, "flag": False, "note": "reliability data unavailable"}

    flagged = score < SUBCONTRACTOR_RISK_THRESHOLD
    note = (
        "below reliability threshold - qualitative risk only, "
        "no monetary estimate (no contract-value input to base one on)"
        if flagged else "within acceptable reliability range"
    )
    return {"score": score, "flag": flagged, "note": note}


# --------------------------------------------------------------------------
# Cost Agent
# --------------------------------------------------------------------------
class CostAgent:

    def __init__(self, task_id):
        self.task_id = task_id
        self.schedule_agent = None
        self.raw_resource = None
        self.equipment_detail = None
        self.cost_breakdown = None
        self.result = None

    def run(self):
        # Reuse schedule_agent's own evaluation instead of re-deriving it.
        self.schedule_agent = ScheduleAgent(task_id=self.task_id)
        self.schedule_agent.run()

        # Raw (pre-delay-evaluation) resource numbers, same source schedule_agent uses.
        self.raw_resource = get_resource_data(self.task_id)
        self.equipment_detail = get_equipment_material_detail(self.task_id)

        delay_cost = compute_delay_cost(self.schedule_agent.combined_result)
        labor_cost = compute_labor_cost(self.raw_resource)
        equipment_cost = compute_equipment_cost(self.equipment_detail)
        material_cost = compute_material_cost(self.raw_resource, self.equipment_detail)
        subcontractor_risk = compute_subcontractor_risk(self.raw_resource)

        self.cost_breakdown = {
            "delay_cost": delay_cost,
            "labor_cost": labor_cost,
            "equipment_cost": equipment_cost,
            "material_cost": material_cost,
            "subcontractor_risk": subcontractor_risk,
        }

        warnings = []
        costed_amounts = []
        for key in ("delay_cost", "labor_cost", "equipment_cost", "material_cost"):
            amt = self.cost_breakdown[key]["amount"]
            if amt is None:
                warnings.append(f"{key} could not be computed - {self.cost_breakdown[key]['basis']}")
            else:
                costed_amounts.append(amt)

        if subcontractor_risk["flag"]:
            warnings.append(f"subcontractor reliability flagged: {subcontractor_risk['note']}")

        if not warnings:
            confidence = "full"
        elif costed_amounts:
            confidence = "partial"
        else:
            confidence = "unknown"

        self.result = {
            "task_id": self.task_id,
            "currency": CURRENCY,
            "cost_breakdown": self.cost_breakdown,
            "total_estimated_cost_impact": round(sum(costed_amounts), 2),
            "data_confidence": confidence,
            "warnings": warnings,
        }
        return self.result

    def display_report(self):
        r = self.result
        zone = TASK_ZONE_MAP.get(self.task_id, self.task_id)
        print("=" * 55)
        print(f"COST AGENT REPORT - Zone: {zone}")
        print("=" * 55)
        for key, label in [
            ("delay_cost", "Delay Cost"),
            ("labor_cost", "Labor Cost"),
            ("equipment_cost", "Equipment Cost"),
            ("material_cost", "Material Cost"),
        ]:
            c = self.cost_breakdown[key]
            amt = f"{c['amount']} {CURRENCY}" if c["amount"] is not None else "N/A"
            print(f"{label:<16}: {amt}")
            print(f"  basis: {c['basis']}")

        risk = self.cost_breakdown["subcontractor_risk"]
        print(f"{'Subcontractor':<16}: score={risk['score']} flagged={risk['flag']}")
        print(f"  note: {risk['note']}")

        print("-" * 55)
        print(f"TOTAL ESTIMATED IMPACT : {r['total_estimated_cost_impact']} {CURRENCY}")
        print(f"DATA CONFIDENCE         : {r['data_confidence']}")
        if r["warnings"]:
            print("WARNINGS:")
            for w in r["warnings"]:
                print(f"  - {w}")
        print("=" * 55)


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    if not TASK_ZONE_MAP:
        print("TASK_ZONE_MAP is empty - add task_id -> Zone entries in schedule_agent.py "
              "before running.")

    all_results = []
    for task_id in TASK_ZONE_MAP:
        agent = CostAgent(task_id=task_id)
        agent.run()
        agent.display_report()
        all_results.append(agent.result)
        print()

    print("=" * 55)
    print("SITE-WIDE COST SUMMARY")
    print("=" * 55)
    site_total = 0.0
    category_totals = {"delay_cost": 0.0, "labor_cost": 0.0, "equipment_cost": 0.0, "material_cost": 0.0}
    for r in all_results:
        site_total += r["total_estimated_cost_impact"]
        for key in category_totals:
            amt = r["cost_breakdown"][key]["amount"]
            if amt is not None:
                category_totals[key] += amt
        zone = TASK_ZONE_MAP.get(r['task_id'], r['task_id'])
        print(f"  {zone:<10} {r['total_estimated_cost_impact']:>10} {CURRENCY:<4} "
              f"(confidence: {r['data_confidence']})")

    print("-" * 55)
    for key, total in category_totals.items():
        print(f"  {key:<16}: {round(total, 2)} {CURRENCY}")
    print(f"  {'SITE TOTAL':<16}: {round(site_total, 2)} {CURRENCY}")
    print("=" * 55)

    # Optional: dump machine-readable output for downstream consumers
    with open("cost_agent_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n[Saved] cost_agent_results.json")
