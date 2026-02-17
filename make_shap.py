import joblib
import shap
import pandas as pd
import numpy as np

# =====================================================
# LOAD TRAINED MODEL + SCALER + FEATURES
# =====================================================

model = joblib.load("baseline_model.pkl")
scaler = joblib.load("baseline_scaler.pkl")
feature_columns = joblib.load("baseline_features.pkl")

print("✅ Model, scaler, and feature list loaded.")

# =====================================================
# LOAD BASELINE DATA (SAME STRUCTURE USED IN TRAINING)
# =====================================================

email_df = pd.read_csv("email_cumulative.csv")
usb_df = pd.read_csv("usb_cumulative.csv")
psy_df = pd.read_csv("psychometric.csv")

email_df.columns = email_df.columns.str.strip()
usb_df.columns = usb_df.columns.str.strip()
psy_df.columns = psy_df.columns.str.strip()

# =====================================================
# AGGREGATION (MUST MATCH TRAINING EXACTLY)
# =====================================================

usb_agg = usb_df.groupby("user").agg(
    sensitive_files_accessed=("sensitive_files_accessed", "sum"),
    files_accessed=("files_accessed", "sum"),
    usb_insertions=("usb_insertions", "sum")
).reset_index()

email_agg = email_df.groupby("user").agg(
    external_emails=("external_emails", "sum"),
    attachments_sent=("attachments_sent", "sum"),
    bcc_in_email=("bcc_in_email", "sum"),
    avg_email_size=("avg_email_size", "mean"),
    total_emails=("total_emails", "sum")
).reset_index()

final_df = usb_agg.merge(email_agg, on="user", how="outer")
final_df.fillna(0, inplace=True)

psy_df = psy_df.rename(columns={"user_id": "user"})

final_df = final_df.merge(
    psy_df[["user", "O", "C", "E", "A", "N"]],
    on="user",
    how="left"
)

final_df.fillna(0, inplace=True)

final_df["C"] = 100 - final_df["C"]
final_df["A"] = 100 - final_df["A"]

# =====================================================
# PREPARE FEATURES
# =====================================================

for col in feature_columns:
    if col not in final_df.columns:
        final_df[col] = 0

X = final_df[feature_columns].copy()
X_scaled = scaler.transform(X)

print("✅ Data prepared for SHAP.")

# =====================================================
# CREATE SHAP EXPLAINER
# =====================================================

explainer = shap.TreeExplainer(
    model,
    feature_perturbation="tree_path_dependent"
)

print("✅ SHAP TreeExplainer created.")

# =====================================================
# OPTIONAL: TEST COMPUTE SHAP VALUES
# =====================================================

shap_values = explainer.shap_values(X_scaled)

print("✅ SHAP values computed successfully.")

# =====================================================
# SAVE EXPLAINER
# =====================================================

joblib.dump(explainer, "shap_explainer.pkl")

print("✅ SHAP explainer saved as shap_explainer.pkl")
