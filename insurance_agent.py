# -*- coding: utf-8 -*-
"""insurance_agent.py

Dynamic Insurance / Claims Agent — designed to plug into the same
multi-agent construction-site ecosystem as complaint_agent.py,
resource_monitoring_agent.py, and weather_agent.py.

Takes a processed complaint/incident (e.g. the output of
DynamicComplaintAgent.process_complaint) and determines policy
applicability, coverage, claimable amount, required documents,
filing deadline, and next actions.
"""

import os
import sqlite3
import datetime
from typing import Dict, Any, List, Optional

# --------------------------------------------------------------------------
# Optional link to the Complaint Agent, so this file can also be run
# standalone against a complaint report dict.
# --------------------------------------------------------------------------
try:
    from complaint_agent import DynamicComplaintAgent
except ImportError:
    DynamicComplaintAgent = None

POLICY_DB_PATH = "insurance_policy.db"


class DynamicInsuranceAgent:
    def __init__(self):
        self.agent_name = "Dynamic-Site-Insurance-Agent"

        # Standard filing turnaround (in days) by severity, used if no
        # policy-specific deadline is found in the DB.
        self.DEFAULT_FILING_DAYS = {
            "CRITICAL": 1,
            "HIGH": 3,
            "MEDIUM": 7,
            "LOW": 14,
        }

    # --------------------------------------------------------------------------
    # 1. DYNAMIC POLICY LOOKUP
    # --------------------------------------------------------------------------
    def _fetch_policy(self, category: str, location: str) -> Dict[str, Any]:
        """Queries a live SQLite policy DB (populated by
        insurance_policy_db_builder.py or a real policy-management system).
        Tries an exact Zone match first, then a Zone="ALL" (site-wide) policy,
        then falls back to hardcoded defaults if nothing is found or the DB
        is unavailable."""
        category_clean = category.strip().lower()

        policy_row = None
        if os.path.exists(POLICY_DB_PATH):
            try:
                import pandas as pd  # local import so pandas isn't a hard dependency
                conn = sqlite3.connect(POLICY_DB_PATH)

                # 1. Exact zone match
                df = pd.read_sql(
                    "SELECT * FROM policies WHERE Category = ? AND Zone = ?",
                    conn,
                    params=(category, location),
                )
                # 2. Site-wide ("ALL") fallback within the DB itself
                if df.empty:
                    df = pd.read_sql(
                        "SELECT * FROM policies WHERE Category = ? AND Zone = 'ALL'",
                        conn,
                        params=(category,),
                    )
                conn.close()
                if not df.empty:
                    policy_row = df.iloc[0].to_dict()
            except Exception:
                policy_row = None

        if policy_row:
            expiry = policy_row.get("Expiry_Date")
            is_expired = False
            if expiry:
                try:
                    is_expired = datetime.datetime.strptime(expiry, "%Y-%m-%d") < datetime.datetime.now()
                except ValueError:
                    is_expired = False

            return {
                "policy_name": policy_row.get("Policy_Name", "Unknown Policy"),
                "policy_number": policy_row.get("Policy_Number", "N/A"),
                "coverage_limit": float(policy_row.get("Coverage_Limit", 0)),
                "deductible": float(policy_row.get("Deductible", 0)),
                "insurer": policy_row.get("Insurer", "Unknown Insurer"),
                "broker_contact": policy_row.get("Broker_Contact", "N/A"),
                "expiry_date": expiry,
                "is_expired": is_expired,
            }

        # ------------------------------------------------------------------
        # Smart fallback defaults by category, if policy DB is offline
        # or no matching row exists.
        # ------------------------------------------------------------------
        if "ppe" in category_clean or "safety" in category_clean:
            return {
                "policy_name": "Workmen's Compensation Policy",
                "policy_number": "WC-DEFAULT",
                "coverage_limit": 2_000_000.0,
                "deductible": 5_000.0,
                "insurer": "Default Insurer Co.",
                "broker_contact": "N/A",
                "expiry_date": None,
                "is_expired": False,
            }
        elif "equipment" in category_clean:
            return {
                "policy_name": "Contractor's Plant & Machinery (CPM) Policy",
                "policy_number": "CPM-DEFAULT",
                "coverage_limit": 5_000_000.0,
                "deductible": 25_000.0,
                "insurer": "Default Insurer Co.",
                "broker_contact": "N/A",
                "expiry_date": None,
                "is_expired": False,
            }
        elif "quality" in category_clean:
            return {
                "policy_name": "Contractor's All Risk (CAR) Policy",
                "policy_number": "CAR-DEFAULT",
                "coverage_limit": 10_000_000.0,
                "deductible": 50_000.0,
                "insurer": "Default Insurer Co.",
                "broker_contact": "N/A",
                "expiry_date": None,
                "is_expired": False,
            }
        else:
            return {
                "policy_name": "General Liability Policy",
                "policy_number": "GL-DEFAULT",
                "coverage_limit": 1_000_000.0,
                "deductible": 10_000.0,
                "insurer": "Default Insurer Co.",
                "broker_contact": "N/A",
                "expiry_date": None,
                "is_expired": False,
            }

    # --------------------------------------------------------------------------
    # 2. DYNAMIC REASONING ENGINE
    # --------------------------------------------------------------------------
    def process_claim(
        self,
        complaint_id: str,
        category: str,
        location: str,
        severity: str,
        root_cause: str,
        estimated_cost_impact: str,
        affected_workers: int = 0,
        incident_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dynamically determines coverage, claim amount, documents, and
        filing deadline for an incident/complaint."""

        category_clean = category.strip().lower()
        severity_clean = severity.strip().upper()

        # Parse the ₹-formatted cost string back into a number
        raw_cost_str = str(estimated_cost_impact).replace("₹", "").replace(",", "").strip()
        try:
            cost_impact = float(raw_cost_str)
        except ValueError:
            cost_impact = 0.0

        # 1. Fetch applicable policy
        policy = self._fetch_policy(category, location)

        # ----------------------------------------------------------------------
        # A. DYNAMIC COVERAGE DETERMINATION
        # ----------------------------------------------------------------------
        if cost_impact <= 0:
            coverage_status = "NOT_APPLICABLE"
        elif cost_impact <= policy["deductible"]:
            coverage_status = "BELOW_DEDUCTIBLE"
        elif cost_impact > policy["coverage_limit"]:
            coverage_status = "PARTIALLY_COVERED"
        else:
            coverage_status = "COVERED"

        # ----------------------------------------------------------------------
        # B. DYNAMIC CLAIM AMOUNT CALCULATION
        # Formula: min(cost_impact, coverage_limit) - deductible, floored at 0
        # ----------------------------------------------------------------------
        capped_cost = min(cost_impact, policy["coverage_limit"])
        claimable_amount = max(0.0, capped_cost - policy["deductible"])
        rounded_claim = int(round(claimable_amount / 500.0) * 500)

        # ----------------------------------------------------------------------
        # C. DYNAMIC FILING DEADLINE
        # ----------------------------------------------------------------------
        days_to_file = self.DEFAULT_FILING_DAYS.get(severity_clean, 7)
        if incident_date:
            try:
                base_date = datetime.datetime.strptime(incident_date, "%Y-%m-%d")
            except ValueError:
                base_date = datetime.datetime.now()
        else:
            base_date = datetime.datetime.now()
        filing_deadline = (base_date + datetime.timedelta(days=days_to_file)).strftime("%Y-%m-%d")

        # ----------------------------------------------------------------------
        # D. DYNAMIC CLAIM PRIORITY
        # ----------------------------------------------------------------------
        if severity_clean == "CRITICAL":
            claim_priority = "URGENT"
        elif severity_clean == "HIGH":
            claim_priority = "HIGH"
        elif severity_clean == "MEDIUM":
            claim_priority = "STANDARD"
        else:
            claim_priority = "LOW"

        # ----------------------------------------------------------------------
        # E. DYNAMIC REQUIRED DOCUMENTS
        # ----------------------------------------------------------------------
        required_documents = ["Incident report", "Cost breakdown / cost impact sheet"]

        if "ppe" in category_clean or "safety" in category_clean:
            required_documents += [
                "Medical report (if injury involved)",
                "Witness statements",
                "Workmen's compensation claim form",
            ]
        elif "equipment" in category_clean:
            required_documents += [
                "Asset purchase invoice / valuation",
                "Maintenance & inspection history",
                "Technician failure diagnosis report",
            ]
        elif "quality" in category_clean:
            required_documents += [
                "Non-Conformance Report (NCR)",
                "Non-destructive testing (NDT) results",
                "Structural engineer remediation report",
            ]
        else:
            required_documents.append("Site supervisor statement")

        # ----------------------------------------------------------------------
        # F. DYNAMIC RECOMMENDED ACTIONS
        # ----------------------------------------------------------------------
        recommended_actions = []

        if policy.get("is_expired"):
            recommended_actions.append(
                f"⚠ Policy {policy['policy_number']} expired on {policy['expiry_date']} — renew immediately before filing."
            )

        if coverage_status == "NOT_APPLICABLE":
            recommended_actions.append("No claim required — cost impact is zero or negligible.")
        elif coverage_status == "BELOW_DEDUCTIBLE":
            recommended_actions.append(
                f"Cost impact is below the ₹{policy['deductible']:,.0f} deductible — absorb internally, do not file."
            )
        else:
            broker_line = f"Notify insurer ({policy['insurer']})"
            if policy.get("broker_contact") and policy["broker_contact"] != "N/A":
                broker_line += f" / broker ({policy['broker_contact']})"
            broker_line += " within 24 hours."
            recommended_actions.append(broker_line)
            recommended_actions.append(f"File claim under policy {policy['policy_number']} before {filing_deadline}.")
            recommended_actions.append("Preserve site evidence (photos, logs, affected materials).")
            if coverage_status == "PARTIALLY_COVERED":
                recommended_actions.append(
                    f"Escalate the ₹{cost_impact - policy['coverage_limit']:,.0f} shortfall above policy limit to management."
                )

        return {
            "complaint_id": complaint_id,
            "category": category,
            "location": location,
            "severity": severity,
            "root_cause": root_cause,
            "policy_name": policy["policy_name"],
            "policy_number": policy["policy_number"],
            "insurer": policy["insurer"],
            "coverage_limit": f"₹{policy['coverage_limit']:,.0f}",
            "deductible": f"₹{policy['deductible']:,.0f}",
            "coverage_status": coverage_status,
            "estimated_cost_impact": f"₹{cost_impact:,.0f}",
            "claimable_amount": f"₹{rounded_claim:,}",
            "claim_priority": claim_priority,
            "filing_deadline": filing_deadline,
            "required_documents": required_documents,
            "recommended_actions": recommended_actions,
            "status": "PENDING_FILING" if coverage_status in ("COVERED", "PARTIALLY_COVERED") else "NO_ACTION",
        }

    # --------------------------------------------------------------------------
    # 3. DISPLAY FORMATTER
    # --------------------------------------------------------------------------
    def display_report(self, report: Dict[str, Any]):
        print(f"CLAIM FOR COMPLAINT ID: {report['complaint_id']}\n")
        print(f"Category: {report['category']}")
        print(f"Location: {report['location']}")
        print(f"Severity: {report['severity']}\n")
        print("Root Cause:")
        print(f"{report['root_cause']}\n")
        print(f"Applicable Policy: {report['policy_name']} ({report['policy_number']})")
        print(f"Insurer: {report['insurer']}")
        print(f"Coverage Limit: {report['coverage_limit']}")
        print(f"Deductible: {report['deductible']}\n")
        print(f"Coverage Status: {report['coverage_status']}")
        print(f"Estimated Cost Impact: {report['estimated_cost_impact']}")
        print(f"Claimable Amount: {report['claimable_amount']}")
        print(f"Claim Priority: {report['claim_priority']}")
        print(f"Filing Deadline: {report['filing_deadline']}\n")
        print("Required Documents:")
        for idx, doc in enumerate(report["required_documents"], 1):
            print(f"{idx}. {doc}")
        print("\nRecommended Actions:")
        for idx, action in enumerate(report["recommended_actions"], 1):
            print(f"{idx}. {action}")
        print(f"\nStatus: {report['status']}")
        print("-" * 60)


# --------------------------------------------------------------------------
# 4. DEMO
# --------------------------------------------------------------------------
if __name__ == "__main__":
    insurance_agent = DynamicInsuranceAgent()

    # If the Complaint Agent is available, chain the two agents together.
    if DynamicComplaintAgent is not None:
        complaint_agent = DynamicComplaintAgent()

        complaint_report = complaint_agent.process_complaint(
            complaint_id="CMP-1026",
            category="Equipment",
            location="Zone A",
            severity="CRITICAL",
            complaint="Tower crane hydraulic pressure failure halting all moment frame lifts."
        )
        print("=== STEP 1: COMPLAINT REPORT ===")
        complaint_agent.display_report(complaint_report)

        claim_report = insurance_agent.process_claim(
            complaint_id=complaint_report["complaint_id"],
            category=complaint_report["category"],
            location=complaint_report["location"],
            severity=complaint_report["severity"],
            root_cause=complaint_report["root_cause"],
            estimated_cost_impact=complaint_report["estimated_cost_impact"],
            affected_workers=complaint_report["affected_workers"],
        )
        print("\n=== STEP 2: INSURANCE CLAIM REPORT ===")
        insurance_agent.display_report(claim_report)

    else:
        # Standalone demo without the Complaint Agent
        claim_report = insurance_agent.process_claim(
            complaint_id="CMP-1027",
            category="Quality",
            location="Zone B",
            severity="MEDIUM",
            root_cause="Non-conformance workmanship defect requiring structural remediation.",
            estimated_cost_impact="₹58,500",
        )
        print("=== STANDALONE: INSURANCE CLAIM REPORT ===")
        insurance_agent.display_report(claim_report)
