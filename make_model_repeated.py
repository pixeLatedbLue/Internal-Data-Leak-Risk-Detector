import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

# =====================================================
# LOAD CUMULATIVE DATA
# =====================================================

email_df = pd.read_csv("email_cumulative.csv")
usb_df = pd.read_csv("usb_cumulative.csv")

# =====================================================
# AGGREGATE
# =====================================================

usb_agg = usb_df.groupby("user").sum().reset_index()
email_agg = email_df.groupby("user").sum().reset_index()

final_df = usb_agg.merge(email_agg, on="user", how="outer")
final_df.fillna(0, inplace=True)

# =====================================================
# FEATURES
# =====================================================

feature_columns = [col for col in final_df.columns if col != "user"]

X = final_df[feature_columns]

# =====================================================
# SCALE
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# TRAIN MODEL
# =====================================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)

# =====================================================
# THRESHOLD
# =====================================================

scores = model.decision_function(X_scaled)
threshold = np.percentile(scores, 5)

# =====================================================
# SAVE EVERYTHING
# =====================================================

joblib.dump(model, "baseline_model.pkl")
joblib.dump(scaler, "baseline_scaler.pkl")
joblib.dump(feature_columns, "baseline_features.pkl")
np.save("relative_threshold.npy", threshold)

print("✅ Model retrained successfully.")
