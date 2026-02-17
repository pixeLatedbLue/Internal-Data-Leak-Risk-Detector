import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime

print("\n🚀 MASTER MULTI-MONTH SIMULATION STARTED\n")

EMAIL_COLUMNS = [
    "user",
    "total_emails",
    "external_emails",
    "attachments_sent",
    "bcc_in_email",
    "avg_email_size"
]

USB_COLUMNS = [
    "user",
    "usb_insertions",
    "files_accessed",
    "sensitive_files_accessed"
]

def initialize_cumulatives():
    pd.DataFrame(columns=EMAIL_COLUMNS).to_csv(
        "email_cumulative.csv", index=False
    )
    pd.DataFrame(columns=USB_COLUMNS).to_csv(
        "usb_cumulative.csv", index=False
    )

initialize_cumulatives()

# =====================================================
# SORT MONTHS CHRONOLOGICALLY
# =====================================================

def parse_month(folder):
    name = folder.replace("_email", "")
    return datetime.strptime(name, "%b_%Y")

email_month_folders = sorted(
    glob("*_email"),
    key=parse_month
)

if not email_month_folders:
    print("❌ No month folders found.")
    exit()

print("📁 Detected Months (Chronological):")
for folder in email_month_folders:
    print("   -", folder)

os.makedirs("threshold_logs", exist_ok=True)
os.makedirs("cumulative_logs", exist_ok=True)
os.makedirs("archived_cumulatives", exist_ok=True)

baseline_exists = False

# =====================================================
# PROCESS MONTHS
# =====================================================

for month_index, email_folder in enumerate(email_month_folders):

    month_label = email_folder.replace("_email", "")
    usb_folder = f"{month_label}_usbfiles"

    print(f"\n📆 PROCESSING MONTH: {month_label.upper()}")

    day = 1

    while True:

        email_file = os.path.join(email_folder, f"email_{day}.csv")
        usb_file = os.path.join(usb_folder, f"usbfile_{day}.csv")

        if not os.path.exists(email_file) or not os.path.exists(usb_file):
            print(f"📅 Month {month_label} complete ({day-1} days)")
            break

        print(f"   📆 Simulating Day {day}")

        shutil.copy(email_file, "daily_email_activity.csv")
        shutil.copy(usb_file, "daily_usb_activity.csv")

        # 🔹 Only run monitor if baseline exists (from previous month)
        if baseline_exists:
            subprocess.run(["python", "monitor.py"])
        else:
            print("   ⏳ Building baseline month (no predictions yet)")

        day += 1

    # =====================================================
    # MONTH END
    # =====================================================

    days_processed = day - 1

    if days_processed > 0:
        print("   🧠 Training / Retraining Baseline Model")
        subprocess.run(["python", "make_model_repeated.py"])
        baseline_exists = True
    else:
        print("   ⚠️ No data processed — skipping training")

    # Archive cumulative
    if os.path.exists("email_cumulative.csv"):
        shutil.move(
            "email_cumulative.csv",
            f"archived_cumulatives/{month_label}_email_cumulative.csv"
        )

    if os.path.exists("usb_cumulative.csv"):
        shutil.move(
            "usb_cumulative.csv",
            f"archived_cumulatives/{month_label}_usb_cumulative.csv"
        )

    # Reset for next month
    initialize_cumulatives()

print("\n🎉 MULTI-MONTH SIMULATION COMPLETE\n")
