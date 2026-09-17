from ultralytics import YOLO
import cv2
import winsound
from datetime import datetime, timedelta
import csv
import os


# Load YOLO model once
model = YOLO("best.pt")


def detect_safety(frame):
    """
    Safety Agent:
    Analyze one camera frame using YOLO and return structured
    safety information for the Worker Protection Agent.
    """
    results = model(frame)
    annotated_frame = results[0].plot()

    detected = []
    worker_count = 0
    helmet_count = 0
    vest_count = 0

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            detected.append(label)

            if label == "Worker":
                worker_count += 1
            elif label == "No-Helmet":
                helmet_count += 1
            elif label == "No-Vest":
                vest_count += 1

    safety_data = {
        "worker_detected": worker_count > 0,
        "worker_count": worker_count,

        "helmet": helmet_count == 0,
        "vest": vest_count == 0,

        "helmet_violation": helmet_count > 0,
        "vest_violation": vest_count > 0,

        "helmet_count": helmet_count,
        "vest_count": vest_count,

        "violation": helmet_count > 0 or vest_count > 0,

        "detected": detected
    }

    return results, annotated_frame, safety_data


def run_safety_monitor():
    """
    Run the Safety Agent independently.
    This keeps the original Safety Agent functionality available.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    os.makedirs("violations", exist_ok=True)

    csv_file = "violations.csv"

    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Time", "Violation"])

    alarm_on = False
    image_saved = False

    last_log_time = datetime.min
    LOG_INTERVAL = timedelta(seconds=10)

    start_time = datetime.now()
    total_screenshots = 0
    total_logs = 0
    last_worker_count = 0
    last_helmet_count = 0
    last_vest_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read camera frame.")
            break

        _, annotated_frame, safety_data = detect_safety(frame)

        detected = safety_data["detected"]
        worker_count = safety_data["worker_count"]
        helmet_count = safety_data["helmet_count"]
        vest_count = safety_data["vest_count"]

        last_worker_count = worker_count
        last_helmet_count = helmet_count
        last_vest_count = vest_count

        violation = safety_data["violation"]

        if safety_data["helmet_violation"]:
            cv2.putText(
                annotated_frame,
                "WARNING: NO HELMET!",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        if safety_data["vest_violation"]:
            cv2.putText(
                annotated_frame,
                "WARNING: NO SAFETY VEST!",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        if violation:
            if not alarm_on:
                winsound.Beep(1000, 500)
                alarm_on = True

            if not image_saved:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"violations/violation_{timestamp}.jpg"

                success = cv2.imwrite(filename, annotated_frame)

                if success:
                    print(f"Screenshot saved: {filename}")
                    total_screenshots += 1
                else:
                    print("Failed to save screenshot!")

                image_saved = True

            if datetime.now() - last_log_time >= LOG_INTERVAL:
                now = datetime.now()
                violation_type = []

                if "No-Helmet" in detected:
                    violation_type.append("No Helmet")

                if "No-Vest" in detected:
                    violation_type.append("No Safety Vest")

                with open(csv_file, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        now.strftime("%d-%m-%Y"),
                        now.strftime("%I:%M:%S %p"),
                        ", ".join(violation_type)
                    ])

                print("Violation logged successfully.")

                total_logs += 1
                last_log_time = now

        else:
            alarm_on = False
            image_saved = False

        cv2.putText(
            annotated_frame,
            f"Workers: {worker_count}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Helmet Violations: {helmet_count}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Vest Violations: {vest_count}",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Total Violations: {helmet_count + vest_count}",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Construction Safety Agent", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            end_time = datetime.now()
            duration = end_time - start_time

            print("\n")
            print("=" * 50)
            print("        CONSTRUCTION SAFETY REPORT")
            print("=" * 50)
            print(
                f"Monitoring Started : "
                f"{start_time.strftime('%d-%m-%Y %I:%M:%S %p')}"
            )
            print(
                f"Monitoring Ended   : "
                f"{end_time.strftime('%d-%m-%Y %I:%M:%S %p')}"
            )
            print(f"Monitoring Time    : {duration}")
            print(f"Workers Detected   : {last_worker_count}")
            print(f"Helmet Violations  : {last_helmet_count}")
            print(f"Vest Violations    : {last_vest_count}")
            print(
                f"Total Violations   : "
                f"{last_helmet_count + last_vest_count}"
            )
            print(f"Screenshots Saved  : {total_screenshots}")
            print(f"CSV Logs Saved     : {total_logs}")

            if last_helmet_count + last_vest_count == 0:
                print("Safety Status      : SAFE")
            else:
                print("Safety Status      : UNSAFE")

            print("=" * 50)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_safety_monitor()
