import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from glob import glob
from datetime import datetime

# =====================================================
# AUTO-DETECT MONTHS
# =====================================================

def parse_month(folder):
    name = folder.replace("_email", "")
    return datetime.strptime(name, "%b_%Y")

email_months = sorted(glob("*_email"), key=parse_month)

if not email_months:
    st.error("No month folders detected.")
    st.stop()

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================

if "month_index" not in st.session_state:
    st.session_state.month_index = 0

if "day" not in st.session_state:
    st.session_state.day = 1

if "baseline_exists" not in st.session_state:
    st.session_state.baseline_exists = False

if "final_df" not in st.session_state:
    st.session_state.final_df = None

if "alerts" not in st.session_state:
    st.session_state.alerts = None

if "processed_day" not in st.session_state:
    st.session_state.processed_day = None

# =====================================================
# CURRENT MONTH INFO
# =====================================================

current_month_folder = email_months[st.session_state.month_index]
current_month = current_month_folder.replace("_email", "")
usb_folder = f"{current_month}_usbfiles"

# =====================================================
# HEADER
# =====================================================

st.title("🔐 Insider Threat Detection Dashboard")
st.markdown(f"### Month: {current_month}")
st.markdown(f"### Day: {st.session_state.day}")

# =====================================================
# SAFE CSV LOADER
# =====================================================

def safe_read(path, cols):
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    if os.path.getsize(path) <= 2:
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path)

EMAIL_COLUMNS = [
    "user","total_emails","external_emails",
    "attachments_sent","bcc_in_email","avg_email_size"
]

USB_COLUMNS = [
    "user","usb_insertions",
    "files_accessed","sensitive_files_accessed"
]

# =====================================================
# NEXT DAY BUTTON
# =====================================================

if st.button("➡️ Next Day"):

    day = st.session_state.day

    email_file = os.path.join(current_month_folder, f"email_{day}.csv")
    usb_file = os.path.join(usb_folder, f"usbfile_{day}.csv")

    # =======================
    # END OF MONTH
    # =======================

    if not os.path.exists(email_file):

        st.info("📅 Month Completed")

        if os.path.exists("email_cumulative.csv"):
            os.system("python make_model_repeated.py")
            st.session_state.baseline_exists = True

        st.session_state.month_index += 1
        st.session_state.day = 1

        if os.path.exists("email_cumulative.csv"):
            os.remove("email_cumulative.csv")
        if os.path.exists("usb_cumulative.csv"):
            os.remove("usb_cumulative.csv")

        st.rerun()

    # =======================
    # PROCESS DAILY DATA
    # =======================

    email_daily = pd.read_csv(email_file)
    usb_daily = pd.read_csv(usb_file)

    email_cum = safe_read("email_cumulative.csv", EMAIL_COLUMNS)
    usb_cum = safe_read("usb_cumulative.csv", USB_COLUMNS)

    email_cum = pd.concat([email_cum, email_daily], ignore_index=True)
    usb_cum = pd.concat([usb_cum, usb_daily], ignore_index=True)

    email_cum.to_csv("email_cumulative.csv", index=False)
    usb_cum.to_csv("usb_cumulative.csv", index=False)

    if not st.session_state.baseline_exists:
        st.session_state.day += 1
        st.warning("⏳ Baseline Month — Accumulating Data Only")
        st.rerun()

    # =======================
    # MONITORING
    # =======================

    model = joblib.load("baseline_model.pkl")
    scaler = joblib.load("baseline_scaler.pkl")
    feature_columns = joblib.load("baseline_features.pkl")
    threshold = np.load("relative_threshold.npy")

    usb_agg = usb_cum.groupby("user").sum().reset_index()
    email_agg = email_cum.groupby("user").sum().reset_index()

    final_df = usb_agg.merge(email_agg, on="user", how="outer")
    final_df.fillna(0, inplace=True)

    for col in feature_columns:
        if col not in final_df.columns:
            final_df[col] = 0

    X = final_df[feature_columns]
    X_scaled = scaler.transform(X)

    scores = model.decision_function(X_scaled)
    final_df["anomaly_score"] = scores

    final_df["FLAG"] = np.where(
        final_df["anomaly_score"] <= threshold,
        "🚨 ALERT",
        "✅ SAFE"
    )

    alerts = final_df[final_df["anomaly_score"] <= threshold]

    # =======================
    # SAVE TO SESSION STATE
    # =======================

    st.session_state.final_df = final_df
    st.session_state.alerts = alerts
    st.session_state.processed_day = day

    st.session_state.day += 1

# =====================================================
# DISPLAY STORED RESULTS (OUTSIDE BUTTON)
# =====================================================

if st.session_state.final_df is not None:

    final_df = st.session_state.final_df
    alerts = st.session_state.alerts
    day = st.session_state.processed_day

    st.success(f"Day {day} Processed")

    col1, col2 = st.columns(2)
    col1.metric("Total Users", len(final_df))
    col2.metric("Alerts Today", len(alerts))

    st.markdown("---")

    # =====================================================
    # FLAGGED USERS + LOAD SAVED SHAP LOGS
    # =====================================================

    st.subheader("🚨 Flagged Users")

    if len(alerts) > 0:

        alerts_sorted = alerts.sort_values("anomaly_score")
        st.dataframe(alerts_sorted)

        selected_user = st.selectbox(
            "Select a flagged user to see explanation:",
            alerts_sorted["user"].values
        )

        LOG_DIR = "daily_shap_logs"
        today_str = f"{current_month}_Day{day}"
        log_file_path = os.path.join(
            LOG_DIR,
            f"shap_log_{today_str}.txt"
        )

        if os.path.exists(log_file_path):

            with open(log_file_path, "r", encoding="utf-8") as file:
                log_content = file.read()

            sections = log_content.split("----------------------------------------")

            user_section = None

            for section in sections:
                if f"User: {selected_user}" in section:
                    user_section = section
                    break

            if user_section:
                st.markdown("### 🔍 Saved SHAP Explanation")
                st.code(user_section)
            else:
                st.warning("No SHAP explanation found for this user.")

        else:
            st.warning("SHAP log file not found for this day.")

    else:
        st.info("No alerts today")

    st.markdown("---")

    st.subheader("📊 All Users")
    st.dataframe(final_df.sort_values("anomaly_score"))

# =====================================================
# RESET BUTTON
# =====================================================

if st.button("🔄 Reset Engine"):

    st.session_state.month_index = 0
    st.session_state.day = 1
    st.session_state.baseline_exists = False
    st.session_state.final_df = None
    st.session_state.alerts = None
    st.session_state.processed_day = None

    if os.path.exists("email_cumulative.csv"):
        os.remove("email_cumulative.csv")
    if os.path.exists("usb_cumulative.csv"):
        os.remove("usb_cumulative.csv")

    st.success("Simulation Reset")
