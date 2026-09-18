from safety_agent import detect_safety
import cv2
import csv
import os
import winsound
from datetime import datetime, timedelta


class WorkerProtectionAgent:

    def __init__(self):
        self.name = "Worker Protection Agent"

        # Hazard / restricted zone.
        # Adjust these coordinates to match the dangerous area in your camera.
        self.zone_x1 = 350
        self.zone_y1 = 100
        self.zone_x2 = 850
        self.zone_y2 = 550

        # Extra area around a worker box used when associating PPE detections.
        self.ppe_margin = 0.20

    def point_inside_hazard_zone(self, x, y):
        return (
            self.zone_x1 <= x <= self.zone_x2
            and self.zone_y1 <= y <= self.zone_y2
        )

    @staticmethod
    def box_iou(box_a, box_b):
        """Calculate IoU between two xyxy boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def expanded_worker_box(self, worker_box):
        x1, y1, x2, y2 = worker_box
        width = x2 - x1
        height = y2 - y1

        return (
            x1 - width * self.ppe_margin,
            y1 - height * self.ppe_margin,
            x2 + width * self.ppe_margin,
            y2 + height * self.ppe_margin
        )

    def _collect_detections(self, results):
        """
        Collect Worker and PPE violation bounding boxes from the exact
        YOLO results returned by Safety Agent.
        """
        workers = []
        no_helmets = []
        no_vests = []

        for result in results:
            names = result.names

            for box in result.boxes:
                cls = int(box.cls[0])
                label = names[cls]

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0]) if box.conf is not None else 0.0

                item = {
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence
                }

                normalized = label.strip().lower().replace("_", "-")

                if normalized == "worker":
                    workers.append(item)

                elif normalized in {
                    "no-helmet",
                    "no helmet",
                    "nohelmet"
                }:
                    no_helmets.append(item)

                elif normalized in {
                    "no-vest",
                    "no vest",
                    "novest"
                }:
                    no_vests.append(item)

        return workers, no_helmets, no_vests

    def analyze_workers(self, results):
        """
        Build a separate protection assessment for every detected worker.

        PPE violations are associated with the worker whose bounding box
        has the strongest spatial overlap with the violation box.
        """
        workers, no_helmets, no_vests = self._collect_detections(results)

        worker_reports = []

        for index, worker in enumerate(workers, start=1):

            worker_box = worker["box"]
            expanded_box = self.expanded_worker_box(worker_box)

            # Find PPE violations that overlap this worker's expanded box.
            helmet_matches = [
                item for item in no_helmets
                if self.box_iou(expanded_box, item["box"]) > 0
            ]

            vest_matches = [
                item for item in no_vests
                if self.box_iou(expanded_box, item["box"]) > 0
            ]

            x1, y1, x2, y2 = worker_box

            # Bottom-center represents where the worker is standing.
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)

            in_hazard_zone = self.point_inside_hazard_zone(
                foot_x,
                foot_y
            )

            helmet_safe = len(helmet_matches) == 0
            vest_safe = len(vest_matches) == 0

            if in_hazard_zone:
                risk = "CRITICAL"
                action = (
                    "STOP WORK IMMEDIATELY. Worker is inside the "
                    "restricted hazard zone."
                )

            elif not helmet_safe and not vest_safe:
                risk = "HIGH"
                action = (
                    "STOP UNSAFE WORK. Worker must wear a safety helmet "
                    "and safety vest before continuing."
                )

            elif not helmet_safe:
                risk = "HIGH"
                action = (
                    "STOP UNSAFE WORK. Worker must wear a safety helmet "
                    "before continuing."
                )

            elif not vest_safe:
                risk = "MEDIUM"
                action = (
                    "Worker must wear a safety vest before continuing."
                )

            else:
                risk = "LOW"
                action = "Worker protection status is OK."

            worker_reports.append({
                "worker_id": index,
                "box": worker_box,
                "helmet": helmet_safe,
                "vest": vest_safe,
                "hazard_zone": in_hazard_zone,
                "risk_level": risk,
                "recommended_action": action,
                "helmet_violations": len(helmet_matches),
                "vest_violations": len(vest_matches)
            })

        return worker_reports

    def assess_worker(self, worker_data, workers_in_hazard_zone=0):
        """
        Keep the original aggregate assessment for compatibility with the
        existing logging/reporting pipeline.
        """
        worker_count = int(worker_data.get("worker_count", 0) or 0)
        helmet_count = int(worker_data.get("helmet_count", 0) or 0)
        vest_count = int(worker_data.get("vest_count", 0) or 0)

        worker_detected = bool(
            worker_data.get("worker_detected", worker_count > 0)
        )

        helmet = bool(
            worker_data.get("helmet", helmet_count == 0)
        )

        vest = bool(
            worker_data.get("vest", vest_count == 0)
        )

        result = {
            "worker_status": "UNKNOWN",
            "risk_level": "UNKNOWN",
            "protection_required": False,
            "recommended_action": "No immediate action required.",
            "safety_analysis_received": True,
            "worker_count": worker_count,
            "helmet_count": helmet_count,
            "vest_count": vest_count,
            "workers_in_hazard_zone": workers_in_hazard_zone,
            "hazard_zone_violation": workers_in_hazard_zone > 0,
        }

        if not worker_detected or worker_count == 0:
            result["worker_status"] = "NO WORKER DETECTED"
            result["risk_level"] = "NONE"
            result["recommended_action"] = (
                "No worker is currently detected. Continue monitoring."
            )
            return result

        result["worker_status"] = "WORKER DETECTED"

        if workers_in_hazard_zone > 0:
            result["risk_level"] = "CRITICAL"
            result["protection_required"] = True
            result["recommended_action"] = (
                "STOP WORK IMMEDIATELY. Worker detected inside the "
                "restricted hazard zone. Move the worker to a safe area."
            )

        elif not helmet and not vest:
            result["risk_level"] = "HIGH"
            result["protection_required"] = True
            result["recommended_action"] = (
                "STOP UNSAFE WORK. The worker must wear a safety helmet "
                "and safety vest before continuing."
            )

        elif not helmet:
            result["risk_level"] = "HIGH"
            result["protection_required"] = True
            result["recommended_action"] = (
                "STOP UNSAFE WORK. The worker must wear a safety helmet "
                "before continuing."
            )

        elif not vest:
            result["risk_level"] = "MEDIUM"
            result["protection_required"] = True
            result["recommended_action"] = (
                "The worker must wear a safety vest before continuing."
            )

        else:
            result["risk_level"] = "LOW"
            result["protection_required"] = False
            result["recommended_action"] = (
                "Worker has the required basic protective equipment "
                "and is outside the configured hazard zone."
            )

        return result


def draw_worker_protection_overlay(
    frame,
    safety_data,
    protection_result,
    agent,
    worker_reports
):
    """Draw hazard zone and individual Worker Protection assessments."""

    risk = protection_result["risk_level"]

    if risk == "CRITICAL":
        status_text = "WORKER PROTECTION: CRITICAL"
        status_color = (0, 0, 255)
    elif risk == "HIGH":
        status_text = "WORKER PROTECTION: HIGH RISK"
        status_color = (0, 0, 255)
    elif risk == "MEDIUM":
        status_text = "WORKER PROTECTION: MEDIUM RISK"
        status_color = (0, 165, 255)
    elif risk == "LOW":
        status_text = "WORKER PROTECTION: LOW RISK"
        status_color = (0, 255, 0)
    else:
        status_text = "WORKER PROTECTION: NO WORKER"
        status_color = (255, 255, 255)

    # Hazard zone
    cv2.rectangle(
        frame,
        (agent.zone_x1, agent.zone_y1),
        (agent.zone_x2, agent.zone_y2),
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "RESTRICTED / HAZARD ZONE",
        (agent.zone_x1, max(25, agent.zone_y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        status_text,
        (20, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        status_color,
        2
    )

    cv2.putText(
        frame,
        f"Workers Detected: {len(worker_reports)}",
        (20, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # Draw each worker's individual assessment.
    for report in worker_reports:
        x1, y1, x2, y2 = map(int, report["box"])

        if report["risk_level"] == "CRITICAL":
            box_color = (0, 0, 255)
        elif report["risk_level"] == "HIGH":
            box_color = (0, 0, 255)
        elif report["risk_level"] == "MEDIUM":
            box_color = (0, 165, 255)
        else:
            box_color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        helmet_text = "OK" if report["helmet"] else "NO"
        vest_text = "OK" if report["vest"] else "NO"
        zone_text = "DANGER" if report["hazard_zone"] else "SAFE"

        label_y = max(20, y1 - 55)

        cv2.putText(
            frame,
            f"W{report['worker_id']} | {report['risk_level']}",
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            box_color,
            2
        )

        cv2.putText(
            frame,
            f"H:{helmet_text} V:{vest_text} Z:{zone_text}",
            (x1, label_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1
        )

    # Summary area
    summary_y = 330

    cv2.putText(
        frame,
        f"Hazard Zone Workers: {protection_result.get('workers_in_hazard_zone', 0)}",
        (20, summary_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 255) if protection_result.get(
            "workers_in_hazard_zone", 0
        ) > 0 else (0, 255, 0),
        2
    )

    action_required = protection_result["protection_required"]

    cv2.putText(
        frame,
        (
            "PROTECTION ACTION REQUIRED"
            if action_required
            else "PROTECTION STATUS: OK"
        ),
        (20, summary_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255) if action_required else (0, 255, 0),
        2
    )

    action = protection_result["recommended_action"]
    if len(action) > 75:
        action = action[:72] + "..."

    cv2.putText(
        frame,
        f"Action: {action}",
        (20, summary_y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1
    )

    return frame


def run_worker_protection():

    agent = WorkerProtectionAgent()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # Evidence screenshots are saved in this folder beside the script's
    # working directory.
    screenshot_dir = os.path.join(
        os.getcwd(),
        "worker_protection_violations"
    )
    os.makedirs(screenshot_dir, exist_ok=True)

    log_file = "worker_protection_log.csv"

    if not os.path.exists(log_file):
        with open(
            log_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Date",
                "Time",
                "Risk Level",
                "Worker Count",
                "Helmet Violations",
                "Vest Violations",
                "Workers In Hazard Zone",
                "Protection Required",
                "Worker Assessments",
                "Recommended Action"
            ])

    start_time = datetime.now()
    last_log_time = datetime.min

    # Used to count an event once when the protection state changes
    # from safe to unsafe.
    last_protection_state = False

    protection_events = 0
    total_logs = 0
    worker_reports = []

    # One alarm and one evidence screenshot per continuous violation event.
    alarm_played_for_event = False
    screenshot_saved_for_event = False

    LOG_INTERVAL = timedelta(seconds=10)

    print("Worker Protection Agent started.")
    print("Hazard-zone protection is enabled.")
    print("Press Q to stop.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read camera frame.")
            break

        # Safety Agent performs YOLO detection.
        results, annotated_frame, safety_data = detect_safety(frame)

        # Build a separate safety assessment for each detected worker.
        worker_reports = agent.analyze_workers(results)

        # Count workers currently inside the restricted zone.
        workers_in_hazard_zone = sum(
            1 for report in worker_reports
            if report["hazard_zone"]
        )

        # Worker Protection Agent makes the aggregate protection decision.
        protection_result = agent.assess_worker(
            safety_data,
            workers_in_hazard_zone
        )

        # If per-worker analysis found a violation that the aggregate Safety
        # Agent summary missed, force protection to the highest worker risk.
        risk_priority = {
            "NONE": 0,
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4,
            "UNKNOWN": 0
        }

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
                protection_result["risk_level"] = highest_worker[
                    "risk_level"
                ]
                protection_result["protection_required"] = (
                    highest_worker["risk_level"] != "LOW"
                )
                protection_result["recommended_action"] = (
                    highest_worker["recommended_action"]
                )

        # Draw individual worker assessments.
        annotated_frame = draw_worker_protection_overlay(
            annotated_frame,
            safety_data,
            protection_result,
            agent,
            worker_reports
        )

        protection_required = protection_result["protection_required"]

        # Detect the start of a new unsafe event.
        new_protection_event = (
            protection_required and not last_protection_state
        )

        if new_protection_event:
            protection_events += 1
            alarm_played_for_event = False
            screenshot_saved_for_event = False

        last_protection_state = protection_required

        if protection_required:

            now = datetime.now()

            # Audible Windows alarm once for each new violation event.
            # Use synchronous Beep first because Windows sound aliases can be
            # disabled/missing on some systems. Three tones make the alert
            # clearly audible.
            if not alarm_played_for_event:
                try:
                    print("\aALARM: Worker protection violation detected!")
                    winsound.Beep(1800, 500)
                    winsound.Beep(1200, 500)
                    winsound.Beep(1800, 700)
                except Exception as alarm_error:
                    print(f"WARNING: Beep alarm failed: {alarm_error}")
                    try:
                        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                    except Exception as message_error:
                        print(f"WARNING: Windows alert failed: {message_error}")

                alarm_played_for_event = True

            # Save one evidence screenshot for each continuous violation.
            if not screenshot_saved_for_event:
                timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                screenshot_path = os.path.join(
                    screenshot_dir,
                    f"worker_protection_{timestamp}.jpg"
                )

                if cv2.imwrite(screenshot_path, annotated_frame):
                    print(
                        f"Protection screenshot saved: {screenshot_path}"
                    )
                else:
                    print(
                        "WARNING: Could not save protection screenshot: "
                        f"{screenshot_path}"
                    )

                screenshot_saved_for_event = True

            # Save a log at most once every 10 seconds while unsafe.
            if now - last_log_time >= LOG_INTERVAL:

                with open(
                    log_file,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        now.strftime("%d-%m-%Y"),
                        now.strftime("%I:%M:%S %p"),
                        protection_result["risk_level"],
                        safety_data.get("worker_count", 0),
                        safety_data.get("helmet_count", 0),
                        safety_data.get("vest_count", 0),
                        workers_in_hazard_zone,
                        protection_required,
                        "; ".join(
                            f"W{r['worker_id']}:"
                            f"Helmet={'OK' if r['helmet'] else 'NO'},"
                            f"Vest={'OK' if r['vest'] else 'NO'},"
                            f"Zone={'DANGER' if r['hazard_zone'] else 'SAFE'},"
                            f"Risk={r['risk_level']}"
                            for r in worker_reports
                        ),
                        protection_result["recommended_action"]
                    ])

                total_logs += 1
                last_log_time = now

        else:
            # The next unsafe event must trigger a fresh alarm and screenshot.
            alarm_played_for_event = False
            screenshot_saved_for_event = False

        cv2.imshow(
            "Worker Protection Agent",
            annotated_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    end_time = datetime.now()
    duration = end_time - start_time

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("Per-worker protection assessment:")
    if worker_reports:
        for report in worker_reports:
            print(
                f"  Worker #{report['worker_id']}: "
                f"Helmet={'OK' if report['helmet'] else 'NOT DETECTED'}, "
                f"Vest={'OK' if report['vest'] else 'NOT DETECTED'}, "
                f"Hazard Zone={'DANGER' if report['hazard_zone'] else 'SAFE'}, "
                f"Risk={report['risk_level']}"
            )
    else:
        print("  No worker detected.")
    print()

    print("=" * 60)
    print("             WORKER PROTECTION REPORT")
    print("=" * 60)
    print(
        f"Monitoring Started : "
        f"{start_time.strftime('%d-%m-%Y %I:%M:%S %p')}"
    )
    print(
        f"Monitoring Ended   : "
        f"{end_time.strftime('%d-%m-%Y %I:%M:%S %p')}"
    )
    print(f"Monitoring Time    : {duration}")
    print(f"Protection Events  : {protection_events}")
    print(f"Logs Saved         : {total_logs}")
    print(f"Log File           : {os.path.abspath(log_file)}")
    print(
        f"Screenshots Folder : {os.path.abspath(screenshot_dir)}"
    )
    print("=" * 60)


if __name__ == "__main__":
    run_worker_protection()
