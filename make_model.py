import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import percentileofscore
import shap
import joblib

# =====================================================
# LOAD DATA (MONTH M)
# =====================================================

email_df = pd.read_csv("email.csv")
usb_df = pd.read_csv("file_usb_activity.csv")
psy_df = pd.read_csv("psychometric.csv")

email_df.columns = email_df.columns.str.strip()
usb_df.columns = usb_df.columns.str.strip()
psy_df.columns = psy_df.columns.str.strip()

print("✅ Training Data Loaded")

# =====================================================
# FEATURE ENGINEERING
# =====================================================

company_domain = "@company.com"

email_df["external_flag"] = email_df["to"].astype(str).str.lower().apply(
    lambda x: 0 if company_domain in x else 1
)

email_features = email_df.groupby("user").agg(
    total_emails=("id", "count"),
    avg_email_size=("size", "mean"),
    attachments_sent=("attachments", "sum"),
    bcc_in_email=("bcc", lambda x: x.notna().sum()),
    external_emails=("external_flag", "sum")
).reset_index()

final_df = email_features.merge(usb_df, on="user", how="left")

psy_df = psy_df.rename(columns={"user_id": "user"})

final_df = final_df.merge(
    psy_df[["user", "O", "C", "E", "A", "N"]],
    on="user",
    how="left"
)

final_df.fillna(0, inplace=True)

# Flip personality logic (lower C/A = higher risk)
final_df["C"] = 100 - final_df["C"]
final_df["A"] = 100 - final_df["A"]

# =====================================================
# WEIGHTS
# =====================================================

weights = {
    "sensitive_files_accessed": 0.18,
    "external_emails": 0.15,
    "attachments_sent": 0.15,
    "bcc_in_email": 0.12,
    "usb_insertions": 0.06,
    "files_accessed": 0.06,
    "avg_email_size": 0.03,
    "total_emails": 0.01,
    "N": 0.07,
    "C": 0.04,
    "A": 0.02,
    "O": 0.01,
    "E": 0.01
}

feature_columns = list(weights.keys())
X = final_df[feature_columns].copy()

for col in feature_columns:
    X[col] *= weights[col]

# =====================================================
# SCALING
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# TRAIN ISOLATION FOREST
# =====================================================

model = IsolationForest(
    n_estimators=300,
    random_state=42
)

model.fit(X_scaled)

scores = model.decision_function(X_scaled)

final_df["anomaly_score"] = scores
final_df["trust_percentile"] = [
    percentileofscore(scores, s) for s in scores
]

# =====================================================
# MONTHLY FLAG = BOTTOM 5% ONLY (SCRIPT 2 LOGIC)
# =====================================================

relative_threshold = np.percentile(scores, 5)

final_df["final_flag"] = (
    final_df["anomaly_score"] < relative_threshold
)

flagged = final_df[final_df["final_flag"]].sort_values("trust_percentile")

print("\n==============================")
print(f"🚨 Total Users: {len(final_df)}")
print(f"🚨 Flagged Users (Bottom 5%): {len(flagged)}")
print(f"🚨 Percentage Flagged: {(len(flagged)/len(final_df))*100:.2f}%")
print("==============================\n")

print(flagged[[
    "user",
    "trust_percentile"
]].head(10))

# =====================================================
# SHAP EXPLANATIONS (ONLY FOR MONTHLY ANOMALIES)
# =====================================================

explainer = shap.TreeExplainer(model)

print("\n🔍 SHAP Explanation for Monthly Flagged Users:\n")

for idx in flagged.index:

    print("\n========================================")
    print("User:", final_df.loc[idx, "user"])
    print("Trust Percentile:",
          round(final_df.loc[idx, "trust_percentile"], 2), "%")

    shap_values = explainer.shap_values(
        X_scaled[idx].reshape(1, -1)
    )

    impacts = pd.DataFrame({
        "Feature": feature_columns,
        "Impact": shap_values[0]
    })

    impacts["AbsImpact"] = impacts["Impact"].abs()
    impacts = impacts.sort_values("AbsImpact", ascending=False)

    print("Top Risk Drivers:")
    for _, row in impacts.head(5).iterrows():
        direction = "↑ Increased Risk" if row["Impact"] < 0 else "↓ Reduced Risk"
        print(f"{row['Feature']} ({direction})")

# =====================================================
# STILL COMPUTE & SAVE DYNAMIC + HARD LIMITS
# (FOR FUTURE DAILY / POLICY USE)
# =====================================================

mean_sensitive = final_df["sensitive_files_accessed"].mean()
std_sensitive = final_df["sensitive_files_accessed"].std()
sensitive_dynamic = mean_sensitive + 3 * std_sensitive

mean_usb = final_df["usb_insertions"].mean()
std_usb = final_df["usb_insertions"].std()
usb_dynamic = mean_usb + 3 * std_usb

HARD_SENSITIVE_LIMIT = 100
HARD_SENSITIVE_RATIO = 0.80
HARD_EXTERNAL_RATIO = 0.75
HARD_USB_LIMIT = 10

# =====================================================
# SAVE BASELINE FOR DAILY MONITORING
# =====================================================

joblib.dump(model, "baseline_model.pkl")
joblib.dump(scaler, "baseline_scaler.pkl")
joblib.dump(feature_columns, "baseline_features.pkl")

np.save("relative_threshold.npy", relative_threshold)
np.save("sensitive_dynamic.npy", sensitive_dynamic)
np.save("usb_dynamic.npy", usb_dynamic)

np.save("hard_sensitive_limit.npy", HARD_SENSITIVE_LIMIT)
np.save("hard_sensitive_ratio.npy", HARD_SENSITIVE_RATIO)
np.save("hard_external_ratio.npy", HARD_EXTERNAL_RATIO)
np.save("hard_usb_limit.npy", HARD_USB_LIMIT)

# =====================================================
# SAVE PER-USER DAILY LIMITS
# =====================================================

user_thresholds = final_df[[
    "user",
    "sensitive_files_accessed",
    "usb_insertions"
]].copy()

user_thresholds["sensitive_limit"] = \
    user_thresholds["sensitive_files_accessed"] * 1.5

user_thresholds["usb_limit"] = \
    user_thresholds["usb_insertions"] * 1.5

user_thresholds = user_thresholds[[
    "user",
    "sensitive_limit",
    "usb_limit"
]]

user_thresholds.to_csv("user_baseline_thresholds.csv", index=False)

print("\n✅ Baseline + Per-User Monitoring Thresholds Saved")
print("✅ Monthly Analysis Completed")
