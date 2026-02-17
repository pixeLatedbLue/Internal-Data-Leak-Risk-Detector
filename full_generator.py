import pandas as pd
import numpy as np
import os
from datetime import datetime
import calendar

# =====================================================
# LOAD USER LIMITS
# =====================================================

limits_df = pd.read_csv("user_baseline_thresholds.csv")
limits_df.columns = limits_df.columns.str.strip()

print(f"✅ Loaded {len(limits_df)} users")

# =====================================================
# SAFE RANDOM
# =====================================================

def safe_randint(low, high):
    if high <= low:
        high = low + 1
    return np.random.randint(low, high)

# =====================================================
# LIMIT-AWARE EMAIL GENERATOR
# =====================================================

def generate_email_behavior(sensitive_limit, usb_limit):

    base_activity = max(5, int(sensitive_limit + usb_limit))

    r = np.random.rand()

    if r < 0.8:
        total_emails = safe_randint(int(0.5 * base_activity),
                                    int(0.75 * base_activity))
    elif r < 0.95:
        total_emails = safe_randint(int(0.75 * base_activity),
                                    int(1.0 * base_activity))
    else:
        total_emails = safe_randint(int(1.0 * base_activity),
                                    int(1.3 * base_activity))

    external_ratio = np.random.uniform(0.05, 0.25)
    attachment_ratio = np.random.uniform(0.05, 0.30)
    bcc_ratio = np.random.uniform(0.0, 0.08)

    external_emails = int(total_emails * external_ratio)
    attachments_sent = int(total_emails * attachment_ratio)
    bcc_in_email = int(total_emails * bcc_ratio)

    avg_email_size = round(np.random.uniform(50, 300), 2)

    return total_emails, external_emails, attachments_sent, bcc_in_email, avg_email_size

# =====================================================
# LIMIT-AWARE USB GENERATOR
# =====================================================

def generate_usb_behavior(sensitive_limit, usb_limit):

    sensitive_limit = max(1, int(round(sensitive_limit)))
    usb_limit = max(1, int(round(usb_limit)))

    r = np.random.rand()

    if r < 0.8:
        sensitive_today = safe_randint(int(0.6 * sensitive_limit),
                                       int(0.85 * sensitive_limit))
        usb_today = safe_randint(int(0.6 * usb_limit),
                                 int(0.85 * usb_limit))

    elif r < 0.95:
        sensitive_today = safe_randint(int(0.85 * sensitive_limit),
                                       int(1.0 * sensitive_limit))
        usb_today = safe_randint(int(0.85 * usb_limit),
                                 int(1.0 * usb_limit))
    else:
        sensitive_today = safe_randint(int(1.0 * sensitive_limit),
                                       int(1.2 * sensitive_limit))
        usb_today = safe_randint(int(1.0 * usb_limit),
                                 int(1.2 * usb_limit))

    files_today = sensitive_today + safe_randint(1, 8)

    return usb_today, files_today, sensitive_today

# =====================================================
# GENERATE 3 MONTHS
# =====================================================

start_date = datetime.now()

for month_offset in range(3):

    year = start_date.year
    month = start_date.month + month_offset

    # Adjust year if month > 12
    if month > 12:
        month -= 12
        year += 1

    month_name = datetime(year, month, 1).strftime("%b").lower()
    month_label = f"{month_name}_{year}"

    num_days = calendar.monthrange(year, month)[1]

    email_folder = f"{month_label}_email"
    usb_folder = f"{month_label}_usbfiles"

    os.makedirs(email_folder, exist_ok=True)
    os.makedirs(usb_folder, exist_ok=True)

    print(f"\n📆 Generating {month_label.upper()} ({num_days} days)")

    for day in range(1, num_days + 1):

        email_rows = []
        usb_rows = []

        for _, row in limits_df.iterrows():

            user = row["user"]
            sensitive_limit = row["sensitive_limit"]
            usb_limit = row["usb_limit"]

            # EMAIL
            total_emails, external_emails, attachments_sent, bcc_in_email, avg_email_size = generate_email_behavior(
                sensitive_limit,
                usb_limit
            )

            email_rows.append({
                "user": user,
                "total_emails": total_emails,
                "external_emails": external_emails,
                "attachments_sent": attachments_sent,
                "bcc_in_email": bcc_in_email,
                "avg_email_size": avg_email_size
            })

            # USB
            usb_today, files_today, sensitive_today = generate_usb_behavior(
                sensitive_limit,
                usb_limit
            )

            usb_rows.append({
                "user": user,
                "usb_insertions": usb_today,
                "files_accessed": files_today,
                "sensitive_files_accessed": sensitive_today
            })

        email_df = pd.DataFrame(email_rows)
        usb_df = pd.DataFrame(usb_rows)

        email_df.to_csv(f"{email_folder}/email_{day}.csv", index=False)
        usb_df.to_csv(f"{usb_folder}/usbfile_{day}.csv", index=False)

    print(f"✅ {month_label.upper()} completed")

print("\n🎉 3 Months Generated Successfully!")
