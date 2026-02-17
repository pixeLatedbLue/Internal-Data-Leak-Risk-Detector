# Insider Threat Dashboard

This project is a Streamlit-powered dashboard and simulation suite for insider-risk monitoring. It ingests daily user activity (USB interactions and email metadata), builds a monthly baseline using anomaly detection, and flags high-risk users with explainable insights using SHAP.

## Overview
- Streamlit UI shows daily KPIs, severity classification, flagged users, and downloadable CSVs.
- Isolation Forest baseline trained monthly on cumulative features; daily scores compare against a relative threshold.
- SHAP explanations identify top feature drivers contributing to a user being flagged.
- Synthetic data generation produces multi-month realistic activity aligned to per-user thresholds.
- CLI engine can simulate multi-month processing, retraining, and SHAP logging.

## Architecture
- Data sources: daily USB activity and email aggregates per user.
- Baseline training: monthly model artifacts and thresholds saved to disk.
- Daily processing: append day’s data into cumulative CSVs, compute anomaly scores, flag users, and render explanations.
- Explainability: SHAP TreeExplainer provides feature-level impact.
- Simulation: generators produce month folders like `feb_2026_email/` and `feb_2026_usbfiles/` with per-day CSVs.

## Key Modules
- [app.py](file:///c:/Users/naval/OneDrive/Desktop/clean_real/app.py): Streamlit dashboard with controls, KPIs, tabs, and SHAP explanations for selected flagged users.
- [monitor.py](file:///c:/Users/naval/OneDrive/Desktop/clean_real/monitor.py): Streamlit variant for step-wise daily processing and SHAP logging to `daily_shap_logs/`.
- [make_model.py](file:///c:/Users/naval/OneDrive/Desktop/clean_real/make_model.py): Monthly baseline training using `email.csv`, `file_usb_activity.csv`, and `psychometric.csv`; applies feature weights and saves artifacts, dynamic/hard limits, and per-user thresholds.
- [make_model_repeated.py](file:///c:/Users/naval/OneDrive/Desktop/clean_real/make_model_repeated.py): Retrains baseline from cumulative aggregates at month end.
- [full_generator.py](file:///c:/Users/naval/OneDrive/Desktop/clean_real/full_generator.py): Generates 3 months of synthetic daily activity from per-user baseline thresholds.
- [engine.py](file:///c:/Users/naval/OneDrive/Desktop/clean_real/engine.py): CLI multi-month orchestrator that copies daily files, runs monitoring, retrains monthly, and archives cumulatives.

## Data Model
- Email features per user: `total_emails`, `external_emails`, `attachments_sent`, `bcc_in_email`, `avg_email_size`.
- USB features per user: `usb_insertions`, `files_accessed`, `sensitive_files_accessed`.
- Psychometric traits (Big Five): `O`, `C`, `E`, `A`, `N` used by baseline training (with `C` and `A` inverted).
- Aggregation: cumulative CSVs summed/averaged as appropriate and merged by `user`.

## Saved Artifacts and Limits
- Model: `baseline_model.pkl`
- Scaler: `baseline_scaler.pkl`
- Feature list: `baseline_features.pkl`
- Thresholds: `relative_threshold.npy`
- Dynamic limits: `sensitive_dynamic.npy`, `usb_dynamic.npy`
- Hard limits: `hard_sensitive_limit.npy`, `hard_sensitive_ratio.npy`, `hard_external_ratio.npy`, `hard_usb_limit.npy`
- Per-user thresholds: [user_baseline_thresholds.csv](file:///c:/Users/naval/OneDrive/Desktop/clean_real/user_baseline_thresholds.csv)

## Project Structure
- Month folders: `feb_2026_email/` and `feb_2026_usbfiles/` contain `email_<day>.csv` and `usbfile_<day>.csv` for each calendar day.
- Daily scratch files: [daily_email_activity.csv](file:///c:/Users/naval/OneDrive/Desktop/clean_real/daily_email_activity.csv), [daily_usb_activity.csv](file:///c:/Users/naval/OneDrive/Desktop/clean_real/daily_usb_activity.csv)
- Cumulative files: [email_cumulative.csv](file:///c:/Users/naval/OneDrive/Desktop/clean_real/email_cumulative.csv), [usb_cumulative.csv](file:///c:/Users/naval/OneDrive/Desktop/clean_real/usb_cumulative.csv)
- Static inputs for baseline training: [email.csv](file:///c:/Users/naval/OneDrive/Desktop/clean_real/email.csv), [file_usb_activity.csv](file:///c:/Users/naval/OneDrive/Desktop/clean_real/file_usb_activity.csv), [psychometric.csv](file:///c:/Users/naval/OneDrive/Desktop/clean_real/psychometric.csv)

## Features
- Month selection and “Next Day” processing with automatic end-of-month retraining.
- Severity classification: Critical/High/Elevated/Normal derived from relative threshold and score quantiles.
- Overview charts: top anomalies, severity distribution, CSV downloads.
- Flagged tab: sortable table, per-user SHAP impact table, human-readable risk explanation, downloadable SHAP CSV.
- All Users tab: filtering by user ID and severity; sorted by anomaly score.
- Reset controls to clear session and cumulative files.

## Getting Started
- Prerequisites: Python 3.10+
- Install dependencies:
  - `pip install streamlit pandas numpy scikit-learn shap joblib`
- Generate per-user thresholds (produced by baseline training):
  - `python make_model.py`
- Generate synthetic months (optional):
  - `python full_generator.py`
- Run the dashboard:
  - `python -m streamlit run app.py`

## Usage Workflow
- Select a month and click “Next Day” to append data and compute anomaly scores.
- First month accumulates baseline (no alerts). Subsequent months score against the trained isolation forest threshold.
- View KPIs and charts in Overview; download alerts or full user scores.
- In Flagged, select a user to see SHAP feature contributions and risk explanation.
- At month end, baseline retrains from cumulative aggregates; session resets to the next month.

## Simulation
- CLI engine:
  - `python engine.py` cycles through detected months, copies daily files, runs monitoring when baseline exists, retrains at month end, archives cumulatives.
- SHAP logging:
  - `monitor.py` writes daily flagged user explanations to `daily_shap_logs/` with top feature drivers.

## Configuration Notes
- Relative threshold is computed as the 5th percentile of decision_function scores in training and reused during monitoring.
- Feature list integrity: monitoring ensures all training features exist, backfilling missing ones with 0.
- Limits files enable policy-like checks and downstream integrations if desired.

## Web3 Integration
- Wallet-based authentication and RBAC:
  - Use wallet sign-in (e.g., MetaMask, WalletConnect) for analysts; derive on-chain roles to gate access to flagged user details and downloads.
- Tamper-evident audit trail:
  - Hash and anchor daily alerts or SHAP logs to a smart contract; store raw CSVs in IPFS/Filecoin; persist CID + hash on-chain for immutable evidence.
- Smart-contract-controlled thresholds:
  - Store `relative_threshold` and limit parameters in a contract; require multi-sig approvals for changes; the app reads current parameters via web3.
- Token-gated access:
  - Restrict sensitive tabs (e.g., SHAP) based on possession of an access NFT or roles in the contract.
- Oracles and external signals:
  - Augment features using oracle-delivered reputational risk scores or breach indicators; periodically pull and include in the feature set.

### Example: Python web3 read/write

```python
from web3 import Web3
import json, numpy as np

w3 = Web3(Web3.HTTPProvider("https://rpc.ankr.com/eth"))
acct = w3.eth.account.from_key(os.environ["ADMIN_PRIVATE_KEY"])

with open("ThresholdsABI.json") as f:
    abi = json.load(f)
contract = w3.eth.contract(address="0xYourContract", abi=abi)

# Read current relative threshold
current = contract.functions.relativeThreshold().call()

# Update threshold (multi-sig or role-protected on-chain)
new_threshold = float(np.load("relative_threshold.npy"))
tx = contract.functions.setRelativeThreshold(new_threshold).build_transaction({
    "from": acct.address,
    "nonce": w3.eth.get_transaction_count(acct.address)
})
signed = acct.sign_transaction(tx)
tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
```

### Example: Anchoring alert digest

```python
import hashlib, pandas as pd
from web3 import Web3

alerts = pd.read_csv("alerts.csv")
digest = hashlib.sha256(alerts.to_csv(index=False).encode()).hexdigest()

contract.functions.recordDailyDigest(digest, "ipfs://<cid>").transact({"from": acct.address})
```

## Deployment
- Local: run Streamlit locally; keep artifacts and month folders in the project root.
- Cloud: containerize and mount persistent storage for artifacts and data folders; set RPC URLs and private keys via environment variables when using web3.

## Troubleshooting
- “No month folders detected”: ensure `*_email/` and `*_usbfiles/` exist; run `full_generator.py` if needed.
- SHAP errors: confirm `baseline_model.pkl`, `baseline_scaler.pkl`, and `baseline_features.pkl` are present and match training.
- Missing data files: first month accumulates baseline; alerts appear only after a baseline exists.
