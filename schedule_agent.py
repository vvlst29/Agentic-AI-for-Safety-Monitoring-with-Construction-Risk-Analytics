
import json
import requests
import pandas as pd

from weather_agent import ConstructionWeatherAgent
from resource_monitoring_agent import DB_PATH, build_database

# Site coordinates used by the real weather agent (default: New York)
SITE_LATITUDE = 19.0760
SITE_LONGITUDE = 72.8777

# Not every task cares about weather (e.g. indoor electrical work).
# Mark which tasks are weather-sensitive here; defaults to True if not listed.
TASK_WEATHER_SENSITIVITY = {
    # "T-101": True,
    # "T-102": False,
}

# The resource simulation DB (labour/equipment/material tables) is keyed by
# Zone, not by schedule task_id, so this map bridges the two agents. Add an
# entry for every task_id you plan to run through the schedule agent.
TASK_ZONE_MAP = {
    "T-101": "Zone A",
    "T-102": "Zone B",
    "T-103": "Zone C",
    "T-104": "Zone D",
}

# The resource monitoring agent doesn't track subcontractor reliability, so
# there's no live source for it yet. Used as a neutral placeholder until one
# exists; it does not come from the resource agent's output.
DEFAULT_SUBCONTRACTOR_RELIABILITY_SCORE = 1.0


def get_weather_data(task_id):
    is_sensitive = TASK_WEATHER_SENSITIVITY.get(task_id, True)

    # schedule_agent calls the weather agent's own analyze_site_risks()
    # and consumes its output directly, instead of re-fetching/re-scoring.
    try:
        weather_agent = ConstructionWeatherAgent(lat=SITE_LATITUDE, lon=SITE_LONGITUDE)
        report = weather_agent.analyze_site_risks()

        if report["status"] == "ERROR":
            raise RuntimeError(
                report["findings"][0] if report["findings"] else "unknown weather error"
            )
    except Exception as e:
        print(f"[Site-Weather-Intelligence-Agent] Connection failed: {e}")
        # Fail safe: treat as weather-sensitive with no data so the
        # schedule agent doesn't silently assume perfect conditions.
        return {
            "task_id": task_id,
            "temperature_c": None,
            "wind_speed_kmh": None,
            "rain_forecast_pct": None,
            "is_weather_sensitive_task": is_sensitive,
            "risk_status": "UNKNOWN",
            "risk_score": None,
            "error": str(e)
        }

    return {
        "task_id": task_id,
        "temperature_c": report["temp"],
        "wind_speed_kmh": report["wind"],
        "rain_forecast_pct": report["rain_prob"],
        "is_weather_sensitive_task": is_sensitive,
        "risk_status": report["status"],
        "risk_score": report["score"]
    }


def get_resource_data(task_id):
    # schedule_agent reads directly from the live resource simulation DB that
    # resource_monitoring_agent.py maintains (labour / equipment / material
    # tables), instead of using mock/hardcoded numbers.
    zone = TASK_ZONE_MAP.get(task_id)

    try:
        # force_reload=False -> reuse the existing DB as-is, picking up
        # whatever the monitoring agent's last simulation cycle wrote.
        conn = build_database(DB_PATH, force_reload=False)
    except Exception as e:
        print(f"[Resource-Monitoring-Agent] Connection failed: {e}")
        return {
            "task_id": task_id,
            "labor_assigned_count": None,
            "labor_required_count": None,
            "equipment_available": None,
            "material_delivery_status": "unknown",
            "subcontractor_reliability_score": None,
            "error": str(e),
        }

    try:
        labour_df = pd.read_sql("SELECT * FROM labour", conn)
        equipment_df = pd.read_sql("SELECT * FROM equipment", conn)
        material_df = pd.read_sql("SELECT * FROM material", conn)
    finally:
        conn.close()

    if zone is None:
        print(f"[Resource-Monitoring-Agent] No zone mapped for task '{task_id}' in "
              f"TASK_ZONE_MAP - using project-wide totals across all zones.")
    else:
        labour_df = labour_df[labour_df["Zone"] == zone]
        equipment_df = equipment_df[equipment_df["Zone"] == zone]
        material_df = material_df[material_df["Zone"] == zone]
        if labour_df.empty and equipment_df.empty and material_df.empty:
            print(f"[Resource-Monitoring-Agent] No records found for zone '{zone}' "
                  f"(task '{task_id}'); check TASK_ZONE_MAP.")

    labor_required = int(labour_df["Required_Labour"].sum())
    labor_assigned = int(labour_df["Available_Labour"].sum())

    equipment_required = int(equipment_df["Required_Quantity"].sum())
    equipment_on_hand = int(equipment_df["Available_Quantity"].sum())
    equipment_available = equipment_on_hand >= equipment_required if equipment_required else True

    material_required = int(material_df["Required_Quantity"].sum())
    material_on_hand = int(material_df["Available_Quantity"].sum())
    if material_required == 0:
        material_delivery_status = "on_time"
    else:
        material_pct = material_on_hand / material_required
        if material_pct >= 1.0:
            material_delivery_status = "on_time"
        elif material_pct >= 0.5:
            material_delivery_status = "pending"
        else:
            material_delivery_status = "delayed"

    return {
        "task_id": task_id,
        "labor_assigned_count": labor_assigned,
        "labor_required_count": labor_required,
        "equipment_available": equipment_available,
        "material_delivery_status": material_delivery_status,   # "on_time" | "delayed" | "pending"
        # Not produced by the resource monitoring agent - see note above.
        "subcontractor_reliability_score": DEFAULT_SUBCONTRACTOR_RELIABILITY_SCORE,
    }


# 2. WEATHER -> DELAY DECISION

def evaluate_weather_delay(weather):
    delay_days = 0
    reasons = []

    if weather.get("temperature_c") is None:
        # Weather fetch failed - can't assess risk, flag it instead of guessing.
        return {
            "task_id": weather["task_id"],
            "will_be_delayed": False,
            "delay_days": 0,
            "reason": f"Weather data unavailable ({weather.get('error', 'unknown error')}) - manual check required",
            "risk_status": "UNKNOWN",
            "risk_score": None
        }

    if weather["is_weather_sensitive_task"] and weather["rain_forecast_pct"] > 50:
        delay_days += 2
        reasons.append(f"High rain forecast ({weather['rain_forecast_pct']}%)")

    if weather["wind_speed_kmh"] > 30:
        delay_days += 1
        reasons.append(f"High wind speed ({weather['wind_speed_kmh']} km/h)")

    if weather["temperature_c"] > 40 or weather["temperature_c"] < 2:
        delay_days += 1
        reasons.append(f"Extreme temperature ({weather['temperature_c']}°C)")

    return {
        "task_id": weather["task_id"],
        "will_be_delayed": delay_days > 0,
        "delay_days": delay_days,
        "reason": "; ".join(reasons) if reasons else "Weather conditions are optimal",
        "risk_status": weather["risk_status"],
        "risk_score": weather["risk_score"]
    }

# 3. RESOURCE -> DELAY DECISION

def evaluate_resource_delay(resource):
    if resource.get("labor_assigned_count") is None:
        # Resource DB read failed - can't assess resource risk, flag it
        # instead of guessing (mirrors the weather agent's fail-safe).
        return {
            "task_id": resource["task_id"],
            "will_be_delayed": False,
            "delay_days": 0,
            "reason": f"Resource data unavailable ({resource.get('error', 'unknown error')}) - manual check required",
        }

    delay_days = 0
    reasons = []

    if resource["labor_assigned_count"] < resource["labor_required_count"]:
        gap = resource["labor_required_count"] - resource["labor_assigned_count"]
        delay_days += gap * 0.5
        reasons.append(f"Labor shortage of {gap} worker(s)")

    if not resource["equipment_available"]:
        delay_days += 1
        reasons.append("Required equipment unavailable")

    if resource["material_delivery_status"] == "delayed":
        delay_days += 1
        reasons.append("Material delivery delayed")
    elif resource["material_delivery_status"] == "pending":
        delay_days += 0.5
        reasons.append("Material delivery status pending confirmation")

    if resource["subcontractor_reliability_score"] < 0.6:
        delay_days += 1
        reasons.append(
            f"Low subcontractor reliability score ({resource['subcontractor_reliability_score']})"
        )

    return {
        "task_id": resource["task_id"],
        "will_be_delayed": delay_days > 0,
        "delay_days": delay_days,
        "reason": "; ".join(reasons) if reasons else "No resource issues"
    }

# 4. SCHEDULE AGENT - combines both

class ScheduleAgent:

    def __init__(self, task_id):
        self.task_id = task_id
        self.weather_result = None
        self.resource_result = None
        self.combined_result = None

    def run(self):
        weather_data = get_weather_data(self.task_id)
        resource_data = get_resource_data(self.task_id)

        self.weather_result = evaluate_weather_delay(weather_data)
        self.resource_result = evaluate_resource_delay(resource_data)

        total_delay = self.weather_result["delay_days"] + self.resource_result["delay_days"]

        self.combined_result = {
            "task_id": self.task_id,
            "will_be_delayed": total_delay > 0,
            "total_delay_days": total_delay,
            "delay_breakdown": {
                "weather_delay_days": self.weather_result["delay_days"],
                "resource_delay_days": self.resource_result["delay_days"]
            }
        }
        return self.combined_result

    def display_report(self):
        zone = TASK_ZONE_MAP.get(self.task_id, self.task_id)
        print("=" * 55)
        print(f"SCHEDULE AGENT REPORT - Zone: {zone}")
        print("=" * 55)

        print("\n--- WEATHER IMPACT ---")
        print(f"Risk Level       : {self.weather_result['risk_status']} ({self.weather_result['risk_score']}/100)")
        print(f"Will be delayed? : {'Yes' if self.weather_result['will_be_delayed'] else 'No'}")
        print(f"Delay (days)     : {self.weather_result['delay_days']}")
        print(f"Reason           : {self.weather_result['reason']}")

        print("\n--- RESOURCE IMPACT ---")
        print(f"Will be delayed? : {'Yes' if self.resource_result['will_be_delayed'] else 'No'}")
        print(f"Delay (days)     : {self.resource_result['delay_days']}")
        print(f"Reason           : {self.resource_result['reason']}")

        print("\n--- COMBINED SCHEDULE IMPACT ---")
        print(f"Overall delayed? : {'Yes' if self.combined_result['will_be_delayed'] else 'No'}")
        print(f"Total delay days : {self.combined_result['total_delay_days']}")
        print("=" * 55)

# 5. EXECUTION

if __name__ == "__main__":
    if not TASK_ZONE_MAP:
        print("TASK_ZONE_MAP is empty - add your task_id -> Zone entries near the top "
              "of this file before running.")

    all_results = []
    for task_id in TASK_ZONE_MAP:
        agent = ScheduleAgent(task_id=task_id)
        agent.run()
        agent.display_report()
        all_results.append(agent.combined_result)
        print()

    print("=" * 55)
    print("SITE-WIDE SUMMARY")
    print("=" * 55)
    for r in all_results:
        zone = TASK_ZONE_MAP.get(r["task_id"], r["task_id"])
        status = "DELAYED" if r["will_be_delayed"] else "on track"
        print(f"  {zone:<10} {status:<10} total delay: {r['total_delay_days']} day(s)")
