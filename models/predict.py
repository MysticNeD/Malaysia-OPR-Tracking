# models/predict.py
"""
Predict next OPR movement (up/same/down) based on trained model using current data.
This is Direction B:
    Input: today (or any date)
    Output: prediction for the next OPR decision in OPR_DECISIONS
Usage:
    python predict.py
"""

from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ------------------------
# Config
# ------------------------
OPR_DECISIONS = [
    "2025-01-01",
    "2025-03-05",
    "2025-05-07",
    "2025-07-09",
    "2025-09-11",
    "2025-11-06",
]
today = datetime.now().date()
OPR_DECISIONS = [d for d in OPR_DECISIONS if datetime.strptime(d, "%Y-%m-%d").date() >= today]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (ROOT / "data").resolve()
MODEL_DIR = (ROOT / "models").resolve()

# ------------------------
# Helper: read CSV if exists
# ------------------------
def read_csv_if_exists(name):
    p = DATA_DIR / name
    if not p.exists():
        print(f"[warn] {p} not found")
        return None
    try:
        df = pd.read_csv(p, dtype=str)
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        return df
    except Exception as e:
        print(f"[warn] failed to read {p}: {e}")
        return None

# ------------------------
# Load trained model
# ------------------------
_model_bundle = None
_clf = None
_features = None

def _load_model():
    global _model_bundle, _clf, _features
    if _model_bundle is None:
        _model_bundle = joblib.load(MODEL_DIR / "model.pkl")
        _clf = _model_bundle["model"]
        _features = _model_bundle["features"]
    return _clf, _features

# ------------------------
# Load historical data
# ------------------------
myor = read_csv_if_exists("myor.csv")
if myor is not None:
    if str(myor.columns[0]).strip().lower() != "date":
        myor = myor.rename(columns={myor.columns[0]: "date"})
    myor_col = next((c for c in myor.columns if c.lower() in ["myor","reference_rate","rate"]), None)
    if myor_col and myor_col != "myor":
        myor = myor.rename(columns={myor_col: "myor"})
    vol_col = next((c for c in myor.columns if "volume" in c.lower() or "aggregate" in c.lower()), None)
    if vol_col and vol_col != "aggregate_volume":
        myor = myor.rename(columns={vol_col: "aggregate_volume"})
    myor["date"] = pd.to_datetime(myor["date"], errors="coerce").dt.date
    myor["myor"] = pd.to_numeric(myor["myor"], errors="coerce")
    myor["aggregate_volume"] = pd.to_numeric(myor.get("aggregate_volume", 0.0), errors="coerce").fillna(0.0)
    myor = myor.dropna(subset=["date"]).reset_index(drop=True)

interbank = read_csv_if_exists("interbank_rates.csv")
if interbank is not None:
    if str(interbank.columns[0]).strip().lower() != "date":
        interbank = interbank.rename(columns={interbank.columns[0]: "date"})
    interbank["date"] = pd.to_datetime(interbank["date"], errors="coerce").dt.date
    if "rate" in interbank.columns:
        interbank["rate"] = pd.to_numeric(interbank["rate"], errors="coerce")
    if "tenor" in interbank.columns:
        interbank["tenor"] = interbank["tenor"].astype(str)
    interbank = interbank.dropna(subset=["date"]).reset_index(drop=True)

interbank_vol = read_csv_if_exists("interbank_volumes.csv")
if interbank_vol is not None:
    if str(interbank_vol.columns[0]).strip().lower() != "date":
        interbank_vol = interbank_vol.rename(columns={interbank_vol.columns[0]: "date"})
    interbank_vol["date"] = pd.to_datetime(interbank_vol["date"], errors="coerce").dt.date
    interbank_vol["volume"] = pd.to_numeric(interbank_vol.get("volume", 0.0), errors="coerce").fillna(0.0)
    interbank_vol = interbank_vol.dropna(subset=["date"]).reset_index(drop=True)

# ------------------------
# Helper: get last OPR date
# ------------------------
def get_last_opr_date(pred_date: date):
    past_oprs = [datetime.strptime(d, "%Y-%m-%d").date() for d in OPR_DECISIONS if datetime.strptime(d, "%Y-%m-%d").date() < pred_date]
    if past_oprs:
        return past_oprs[-1]
    return pred_date - timedelta(days=60)

# ------------------------
# Feature generator (Direction B)
# ------------------------
def generate_features(pred_date: date):
    """
    Generate features for prediction using pred_date.
    Use data from last OPR to today
    """
    feat = {"date": pred_date}

    lb_start = get_last_opr_date(pred_date)
    lb_end = datetime.now().date()

    # MYOR features
    if myor is not None:
        window = myor[(myor["date"] > lb_start) & (myor["date"] <= lb_end)]
        if not window.empty:
            feat["myor_mean_7d"] = window["myor"].mean()
            feat["myor_last"] = window["myor"].iloc[-1]
            feat["myor_vol_mean_7d"] = window["aggregate_volume"].mean()
            feat["myor_vol_last"] = window["aggregate_volume"].iloc[-1]
        else:
            feat["myor_mean_7d"] = 0.0
            feat["myor_last"] = 0.0
            feat["myor_vol_mean_7d"] = 0.0
            feat["myor_vol_last"] = 0.0

    # Interbank overnight
    if interbank is not None:
        ov = interbank[(interbank["tenor"].str.lower().str.contains("overnight", na=False)) &
                       (interbank["date"] > lb_start) & (interbank["date"] <= lb_end)]
        feat["overnight_mean_7d"] = ov["rate"].mean() if not ov.empty else 0.0
        feat["overnight_last"] = ov["rate"].iloc[-1] if not ov.empty else 0.0

        # 1-month
        m1 = interbank[(interbank["tenor"].str.contains("1_month|1 month|1_month", case=False, regex=True)) &
                       (interbank["date"] > lb_start) & (interbank["date"] <= lb_end)]
        feat["m1_mean_7d"] = m1["rate"].mean() if not m1.empty else 0.0
    else:
        feat["overnight_mean_7d"] = 0.0
        feat["overnight_last"] = 0.0
        feat["m1_mean_7d"] = 0.0

    # Interbank volume
    if interbank_vol is not None:
        vv = interbank_vol[(interbank_vol["date"] > lb_start) & (interbank_vol["date"] <= lb_end)]
        feat["vol_mean_7d"] = vv["volume"].mean() if not vv.empty else 0.0
        feat["vol_sum_7d"] = vv["volume"].sum() if not vv.empty else 0.0
    else:
        feat["vol_mean_7d"] = 0.0
        feat["vol_sum_7d"] = 0.0

    # Spread
    feat["myor_minus_opr"] = 0.0  # unknown next OPR

    # Diff features
    feat["myor_diff"] = feat["myor_last"] - feat["myor_mean_7d"]
    feat["overnight_diff"] = feat["overnight_last"] - feat["overnight_mean_7d"]

    # Build DataFrame
    clf, _features = _load_model()
    X_pred = pd.DataFrame([feat])
    for f in _features:
        if f not in X_pred.columns or pd.isna(X_pred.at[0, f]):
            X_pred[f] = 0.0

    return X_pred[_features]

# ------------------------
# Predict
# ------------------------
def predict_opr(pred_date):
    clf, _ = _load_model()
    if isinstance(pred_date, str):
        pred_date_dt = datetime.strptime(pred_date, "%Y-%m-%d").date()
    else:
        pred_date_dt = pred_date
    X_pred = generate_features(pred_date_dt)
    label = clf.predict(X_pred)[0]
    proba = clf.predict_proba(X_pred)[0]
    return label, dict(zip(clf.classes_, proba))

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    print("[info] Predicting for next OPR decision dates based on today's data...")

    if not OPR_DECISIONS:
        print("[warning] No future OPR decision dates found.")
    else:
        for fd in OPR_DECISIONS:
            label, proba = predict_opr(fd)
            proba = {k: float(v) for k, v in proba.items()}
            print(f"Today: {today}, Next OPR date: {fd}")
            print(f"Predicted OPR movement: {label.upper()}, Probabilities: {proba}")
