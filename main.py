from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from weather_agent import ConstructionWeatherAgent
import resource_monitoring_agent as rma
from schedule_agent import ScheduleAgent, TASK_ZONE_MAP
from cost_agent import CostAgent, CURRENCY
from insurance_agent import DynamicInsuranceAgent
from safety_agent import detect_safety
from worker_protection import WorkerProtectionAgent

try:
    from complaint_agent import DynamicComplaintAgent
except ImportError:
    DynamicComplaintAgent = None

# Sample incidents used to drive the complaint + insurance agents until
# they're wired up to a live intake feed (e.g. a form, ticketing system,
# or IoT/sensor triggers per zone).
SAMPLE_INCIDENTS = [
    {
        "complaint_id": "CMP-1025",
        "category": "PPE/Safety",
        "location": "Zone C",
        "severity": "HIGH",
        "complaint": "Safety gloves unavailable for workers.",
    },
    {
        "complaint_id": "CMP-1026",
        "category": "Equipment",
        "location": "Zone A",
        "severity": "CRITICAL",
        "complaint": "Tower crane hydraulic pressure failure halting all moment frame lifts.",
    },
    {
        "complaint_id": "CMP-1027",
        "category": "Quality",
        "location": "Zone B",
        "severity": "MEDIUM",
        "complaint": "Honeycombing and voids detected on concrete shear wall after formwork removal.",
    },
]

app = FastAPI(title="Construction Site Risk API")

# Serve the dashboard from the same FastAPI server. This avoids browser
# connection/CORS problems caused by opening the HTML file directly.
FRONTEND_FILE = Path(__file__).with_name("Frontend.html")

@app.get("/", include_in_schema=False)
def serve_frontend():
    if not FRONTEND_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend HTML file not found.")
    return FileResponse(
        FRONTEND_FILE,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )

# Allow the React dev server to call this API.
# Update the origin(s) once you know your frontend's deployed URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SiteRiskResponse(BaseModel):
    status: str
    score: int
    temp: float | None = None
    wind: float | None = None
    rain_prob: float | None = None
    findings: list[str] = []
    actions: list[str] = []


@app.get("/api/site-risk", response_model=SiteRiskResponse)
def get_site_risk(lat: float = 19.0760, lon: float = 72.8777):
    """
    Runs the ConstructionWeatherAgent for a given site (lat/lon)
    and returns its risk assessment as JSON.
    """
    agent = ConstructionWeatherAgent(lat=lat, lon=lon)
    result = agent.analyze_site_risks()

    if result["status"] == "ERROR":
        raise HTTPException(status_code=502, detail=result["findings"][0])

    return result


@app.get("/api/resource-dashboard")
def get_resource_dashboard(refresh: bool = True):
    """
    Runs the ResourceMonitoringAgent's logic: loads/reuses the simulated
    database, optionally advances one simulation cycle, and returns the
    labour/equipment/material metrics as JSON.
    """
    conn = rma.build_database()

    if refresh:
        rma.simulate_labour_changes(conn)
        rma.simulate_equipment_changes(conn)
        rma.simulate_material_changes(conn)

    labour = rma.compute_labour_metrics(conn)
    equipment = rma.compute_equipment_metrics(conn)
    material = rma.compute_material_metrics(conn)
    conn.close()

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "labour": labour,
        "equipment": equipment,
        "material": material,
    }


@app.get("/api/schedule-report")
def get_schedule_report():
    """
    Runs the ScheduleAgent for every task in TASK_ZONE_MAP, combining live
    weather risk and live resource data per zone, plus a site-wide summary.
    """
    if not TASK_ZONE_MAP:
        raise HTTPException(
            status_code=500,
            detail="TASK_ZONE_MAP is empty in schedule_agent.py - add task_id -> Zone entries.",
        )

    tasks = []
    for task_id, zone in TASK_ZONE_MAP.items():
        agent = ScheduleAgent(task_id=task_id)
        agent.run()
        tasks.append({
            "task_id": task_id,
            "zone": zone,
            "weather": agent.weather_result,
            "resource": agent.resource_result,
            "combined": agent.combined_result,
        })

    summary = [
        {
            "zone": t["zone"],
            "delayed": t["combined"]["will_be_delayed"],
            "total_delay_days": t["combined"]["total_delay_days"],
        }
        for t in tasks
    ]

    return {"tasks": tasks, "summary": summary}


@app.get("/api/cost-report")
def get_cost_report():
    """
    Runs the CostAgent for every task in TASK_ZONE_MAP: delay/labor/equipment/
    material cost breakdown per zone, plus a site-wide cost summary.
    """
    if not TASK_ZONE_MAP:
        raise HTTPException(
            status_code=500,
            detail="TASK_ZONE_MAP is empty in schedule_agent.py - add task_id -> Zone entries.",
        )

    results = []
    for task_id in TASK_ZONE_MAP:
        agent = CostAgent(task_id=task_id)
        agent.run()
        results.append(agent.result)

    site_total = 0.0
    category_totals = {"delay_cost": 0.0, "labor_cost": 0.0, "equipment_cost": 0.0, "material_cost": 0.0}
    zone_summary = []

    for r in results:
        site_total += r["total_estimated_cost_impact"]
        for key in category_totals:
            amt = r["cost_breakdown"][key]["amount"]
            if amt is not None:
                category_totals[key] += amt
        zone_summary.append({
            "zone": TASK_ZONE_MAP.get(r["task_id"], r["task_id"]),
            "total_estimated_cost_impact": r["total_estimated_cost_impact"],
            "data_confidence": r["data_confidence"],
        })

    return {
        "currency": CURRENCY,
        "tasks": results,
        "zone_summary": zone_summary,
        "category_totals": {k: round(v, 2) for k, v in category_totals.items()},
        "site_total": round(site_total, 2),
    }


@app.get("/api/complaint-report")
def get_complaint_report():
    """
    Runs the DynamicComplaintAgent for every incident in SAMPLE_INCIDENTS,
    pulling live labour/material/equipment data (via resource_monitoring_agent)
    and live weather risk (via ConstructionWeatherAgent) into each report.
    """
    if DynamicComplaintAgent is None:
        raise HTTPException(
            status_code=500,
            detail="complaint_agent.py could not be imported - check it's present alongside main.py.",
        )

    complaint_agent = DynamicComplaintAgent()
    complaints = [
        complaint_agent.process_complaint(
            complaint_id=incident["complaint_id"],
            category=incident["category"],
            location=incident["location"],
            severity=incident["severity"],
            complaint=incident["complaint"],
        )
        for incident in SAMPLE_INCIDENTS
    ]

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "complaints": complaints,
    }


@app.get("/api/insurance-report")
def get_insurance_report():
    """
    Runs the DynamicInsuranceAgent for each sample/live incident.

    If complaint_agent.py is available, each incident in SAMPLE_INCIDENTS is
    first passed through DynamicComplaintAgent.process_complaint() (so root
    cause, affected workers, and cost impact are generated dynamically),
    then the resulting complaint report is fed into insurance_agent's
    process_claim(). Otherwise falls back to a static demo claim so the tab
    still has something to show.
    """
    insurance_agent = DynamicInsuranceAgent()
    claims = []

    if DynamicComplaintAgent is not None:
        complaint_agent = DynamicComplaintAgent()
        for incident in SAMPLE_INCIDENTS:
            complaint_report = complaint_agent.process_complaint(
                complaint_id=incident["complaint_id"],
                category=incident["category"],
                location=incident["location"],
                severity=incident["severity"],
                complaint=incident["complaint"],
            )
            claim_report = insurance_agent.process_claim(
                complaint_id=complaint_report["complaint_id"],
                category=complaint_report["category"],
                location=complaint_report["location"],
                severity=complaint_report["severity"],
                root_cause=complaint_report["root_cause"],
                estimated_cost_impact=complaint_report["estimated_cost_impact"],
                affected_workers=complaint_report.get("affected_workers", 0),
            )
            claims.append(claim_report)
    else:
        # Static fallback matching the last known complaint_agent output,
        # so the tab still renders something useful without that module.
        claims.append(insurance_agent.process_claim(
            complaint_id="CMP-1026",
            category="Equipment",
            location="Zone A",
            severity="CRITICAL",
            root_cause="Equipment breakdown and unplanned maintenance (36 unit(s) down).",
            estimated_cost_impact="₹646,500",
            affected_workers=406,
        ))

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "claims": claims,
    }


@app.post("/api/safety-report")
async def get_safety_report(file: UploadFile = File(...)):
    """
    Runs the Safety Agent on one uploaded image/frame.

    The React frontend can capture a camera frame and send it as an image.
    Returns structured JSON containing worker, helmet, and vest detection data.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Please upload an image. Received content type: {file.content_type or 'unknown'}"
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    import numpy as np
    import cv2

    image_array = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image."
        )

    try:
        _, _, safety_data = detect_safety(frame)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Safety Agent failed: {exc}"
        )

    total_violations = (
        safety_data.get("helmet_count", 0)
        + safety_data.get("vest_count", 0)
    )

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "UNSAFE" if safety_data.get("violation") else "SAFE",
        "worker_detected": safety_data.get("worker_detected", False),
        "worker_count": safety_data.get("worker_count", 0),
        "helmet": safety_data.get("helmet", True),
        "vest": safety_data.get("vest", True),
        "helmet_violation": safety_data.get("helmet_violation", False),
        "vest_violation": safety_data.get("vest_violation", False),
        "helmet_count": safety_data.get("helmet_count", 0),
        "vest_count": safety_data.get("vest_count", 0),
        "total_violations": total_violations,
        "violation": safety_data.get("violation", False),
        "detected": safety_data.get("detected", []),
    }


@app.post("/api/worker-protection-report")
async def get_worker_protection_report(file: UploadFile = File(...)):
    """
    Runs Safety Agent + Worker Protection Agent on one uploaded image/frame.

    Worker Protection performs per-worker assessment including helmet,
    vest, restricted hazard-zone status, risk level, and recommended action.
    """
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    import numpy as np
    import cv2

    image_array = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image."
        )

    try:
        results, _, safety_data = detect_safety(frame)

        agent = WorkerProtectionAgent()
        worker_reports = agent.analyze_workers(results)

        workers_in_hazard_zone = sum(
            1
            for report in worker_reports
            if report["hazard_zone"]
        )

        protection_result = agent.assess_worker(
            safety_data,
            workers_in_hazard_zone
        )

        risk_priority = {
            "NONE": 0,
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4,
            "UNKNOWN": 0,
        }

        # Keep the aggregate result consistent with the highest
        # individual worker risk.
        if worker_reports:
            highest_worker = max(
                worker_reports,
                key=lambda item: risk_priority.get(
                    item["risk_level"], 0
                )
            )

            if risk_priority.get(
                highest_worker["risk_level"], 0
            ) > risk_priority.get(
                protection_result["risk_level"], 0
            ):
                protection_result["risk_level"] = (
                    highest_worker["risk_level"]
                )
                protection_result["protection_required"] = (
                    highest_worker["risk_level"] != "LOW"
                )
                protection_result["recommended_action"] = (
                    highest_worker["recommended_action"]
                )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Worker Protection Agent failed: {exc}"
        )

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "safety": {
            "status": (
                "UNSAFE"
                if safety_data.get("violation")
                else "SAFE"
            ),
            "worker_detected": safety_data.get(
                "worker_detected", False
            ),
            "worker_count": safety_data.get("worker_count", 0),
            "helmet_violation": safety_data.get(
                "helmet_violation", False
            ),
            "vest_violation": safety_data.get(
                "vest_violation", False
            ),
            "helmet_count": safety_data.get("helmet_count", 0),
            "vest_count": safety_data.get("vest_count", 0),
            "detected": safety_data.get("detected", []),
        },
        "worker_protection": protection_result,
        "worker_reports": worker_reports,
        "workers_in_hazard_zone": workers_in_hazard_zone,
    }


@app.get("/health")
def health_check():
    return {"ok": True}
