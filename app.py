import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from glob import glob
from datetime import datetime
import shap

st.set_page_config(page_title="Insider Threat Dashboard", page_icon="🔐", layout="wide")
st.markdown("""
<style>
:root{--primary:#0F766E;--bg:#0b1324;--text:#e2e8f0;--card:#111827;--accent:#38bdf8;--danger:#ef4444;--success:#22c55e}
.stApp{background-color:var(--bg)}
h1,h2,h3,h4,h5,p,span{color:var(--text)!important}
.block-container{padding-top:1rem}
.header{display:flex;justify-content:space-between;align-items:center;padding:4px 0 18px;border-bottom:1px solid #1f2937;margin-bottom:14px}
.header .title{font-size:22px;font-weight:700;color:#e2e8f0}
.header .badge{background:var(--primary);color:#fff;padding:6px 10px;border-radius:999px;font-size:12px}
.kpi{display:flex;gap:16px;margin-bottom:12px}
.kpi .metric{flex:1;background:var(--card);padding:14px;border-radius:10px;border:1px solid #1f2937}
.kpi .metric h3{margin:0;font-size:13px;color:#94a3b8}
.kpi .metric .value{font-size:24px;color:#e2e8f0;font-weight:600}
.stTabs [role='tablist'] button{background:var(--card);border-radius:8px}
.stMarkdown,.stDataFrame{background:transparent}
</style>
""", unsafe_allow_html=True)

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
# SESSION STATE INIT
# =====================================================

for key in ["month_index", "day", "baseline_exists",
            "final_df", "alerts", "X_scaled"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if key in ["month_index", "day"] else None

if st.session_state.day == 0:
    st.session_state.day = 1

# =====================================================
# CURRENT MONTH
# =====================================================

current_month_folder = email_months[st.session_state.month_index]
current_month = current_month_folder.replace("_email", "")
usb_folder = f"{current_month}_usbfiles"

st.sidebar.title("Controls")
month_labels = [f.replace("_email", "") for f in email_months]
selected_month = st.sidebar.selectbox("Month", month_labels, index=st.session_state.month_index)
if selected_month != current_month:
    new_index = month_labels.index(selected_month)
    st.session_state.month_index = new_index
    st.session_state.day = 1
    st.session_state.final_df = None
    st.session_state.alerts = None
    st.session_state.X_scaled = None
    if os.path.exists("email_cumulative.csv"):
        os.remove("email_cumulative.csv")
    if os.path.exists("usb_cumulative.csv"):
        os.remove("usb_cumulative.csv")
    st.experimental_rerun()

st.markdown(f"<div class='header'><div class='title'>Insider Risk Dashboard</div><div class='badge'>Month: {current_month} • Day {st.session_state.day}</div></div>", unsafe_allow_html=True)

# =====================================================
# SAFE LOADER
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

st.sidebar.markdown("---")
next_day = st.sidebar.button("➡️ Next Day")
reset_engine_sidebar = st.sidebar.button("🔄 Reset Engine")

if next_day:

    day = st.session_state.day

    email_file = os.path.join(current_month_folder, f"email_{day}.csv")
    usb_file = os.path.join(usb_folder, f"usbfile_{day}.csv")

    if not os.path.exists(email_file):

        st.info("📅 Month Completed — retraining if data exists")

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

    email_daily = pd.read_csv(email_file)
    usb_daily = pd.read_csv(usb_file)

    email_cum = safe_read("email_cumulative.csv", EMAIL_COLUMNS)
    usb_cum = safe_read("usb_cumulative.csv", USB_COLUMNS)

    email_cum = pd.concat([email_cum, email_daily])
    usb_cum = pd.concat([usb_cum, usb_daily])

    email_cum.to_csv("email_cumulative.csv", index=False)
    usb_cum.to_csv("usb_cumulative.csv", index=False)

    if not st.session_state.baseline_exists:
        st.session_state.day += 1
        st.warning("⏳ Baseline Month — Accumulating Data Only")
        st.rerun()

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

    st.session_state.final_df = final_df
    st.session_state.alerts = alerts
    st.session_state.X_scaled = X_scaled

    st.session_state.day += 1

# =====================================================
# DISPLAY RESULTS
# =====================================================

if st.session_state.final_df is not None:

    final_df = st.session_state.final_df
    alerts = st.session_state.alerts
    X_scaled = st.session_state.X_scaled
    threshold = np.load("relative_threshold.npy")
    if len(final_df) > 0:
        q25 = float(final_df["anomaly_score"].quantile(0.25))
        q10 = float(final_df["anomaly_score"].quantile(0.10))
        def _sev(s):
            if s <= float(threshold):
                return "Critical"
            elif s <= q10:
                return "High"
            elif s <= q25:
                return "Elevated"
            else:
                return "Normal"
        final_df["severity"] = final_df["anomaly_score"].apply(_sev)
    else:
        final_df["severity"] = "Normal"

    st.success("Day processed")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", len(final_df))
    col2.metric("Critical Alerts", int((final_df["severity"]=="Critical").sum()))
    col3.metric("At Risk (High+Elevated)", int((final_df["severity"].isin(["High","Elevated"])).sum()))
    col4.metric("Min Score", float(final_df["anomaly_score"].min()))

    tabs = st.tabs(["Overview", "Flagged", "All Users"])

    with tabs[0]:
        top_n = st.slider("Top N anomalies", 5, 30, 10)
        top_df = final_df.sort_values("anomaly_score").head(top_n)[["user", "anomaly_score"]]
        st.bar_chart(top_df.set_index("user"))
        sev_counts = final_df["severity"].value_counts().reindex(["Critical","High","Elevated","Normal"]).fillna(0)
        st.bar_chart(sev_counts)
        c1, c2 = st.columns(2)
        with c1:
            if len(alerts) > 0:
                st.download_button("Download alerts CSV", alerts.sort_values("anomaly_score").to_csv(index=False), "alerts.csv", "text/csv")
        with c2:
            st.download_button("Download all users CSV", final_df.sort_values("anomaly_score").to_csv(index=False), "all_users.csv", "text/csv")

    with tabs[1]:
        if len(alerts) > 0:

            alerts_sorted = alerts.sort_values("anomaly_score")
            st.subheader("Flagged Users")
            st.dataframe(alerts_sorted)

            selected_user = st.selectbox(
                "Select a flagged user to see explanation:",
                alerts_sorted["user"].values
            )

            model = joblib.load("baseline_model.pkl")
            feature_columns = joblib.load("baseline_features.pkl")

            user_index = final_df.index[
                final_df["user"] == selected_user
            ][0]

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled[user_index].reshape(1, -1))[0]

            shap_df = pd.DataFrame({
                "Feature": feature_columns,
                "SHAP Value": shap_values
            }).sort_values(
                by="SHAP Value",
                key=abs,
                ascending=False
            )

            st.markdown("SHAP Feature Contributions")
            st.dataframe(shap_df)
            st.download_button("Download SHAP CSV", shap_df.to_csv(index=False), f"{selected_user}_shap.csv", "text/csv")

            st.markdown("Risk Explanation")

            top_features = shap_df.head(3)

            st.write(
                f"User {selected_user} flagged due to:"
            )

            for _, row in top_features.iterrows():
                direction = "increased risk" if row["SHAP Value"] < 0 else "reduced risk"
                st.write(
                    f"- {row['Feature']} {direction} (impact: {row['SHAP Value']:.4f})"
                )

            total_impact = shap_df["SHAP Value"].sum()

            if total_impact < 0:
                st.error("Overall behaviour strongly deviates from baseline")
            else:
                st.success("Deviation appears limited")

            st.info("Negative SHAP values push toward being flagged; positive toward normal")

        else:
            st.info("No alerts today.")

    with tabs[2]:
        st.subheader("All Users")
        query = st.text_input("Filter by user ID contains")
        sev_filter = st.multiselect("Severity filter", ["Critical","High","Elevated","Normal"], default=["Critical","High","Elevated","Normal"])
        display_df = final_df.sort_values("anomaly_score")
        if query:
            display_df = display_df[display_df["user"].astype(str).str.contains(query, case=False)]
        if sev_filter:
            display_df = display_df[display_df["severity"].isin(sev_filter)]
        st.dataframe(display_df, use_container_width=True)

else:
    st.info("Click Next Day to process data")

# =====================================================
# RESET
# =====================================================

if reset_engine_sidebar:

    st.session_state.clear()

    if os.path.exists("email_cumulative.csv"):
        os.remove("email_cumulative.csv")
    if os.path.exists("usb_cumulative.csv"):
        os.remove("usb_cumulative.csv")

    st.success("Simulation Reset")
