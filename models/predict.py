# models/predict.py
OPR_DECISIONS = ["2025-09-04", "2025-11-06"]
"""
Predict next OPR movement (up/same/down) based on trained model.
Usage:
    python predict.py
"""

from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import joblib
import warnings

today = datetime.now().date()
OPR_DECISIONS = [d for d in OPR_DECISIONS if datetime.strptime(d, "%Y-%m-%d").date() >= today]

warnings.filterwarnings("ignore")

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
# ------------------------
# 延迟加载模型
# ------------------------
clf = None
features = None

def load_model():
    global clf, features
    import joblib
    MODEL_DIR = Path(__file__).resolve().parent
    model_path = MODEL_DIR / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Please train the model first.")
    model_bundle = joblib.load(model_path)
    clf = model_bundle["model"]
    features = model_bundle["features"]

def predict_opr(pred_date: str):
    global clf, features
    if clf is None or features is None:
        load_model()
    from datetime import datetime
    X_pred = generate_features(datetime.strptime(pred_date, "%Y-%m-%d").date())
    label = clf.predict(X_pred)[0]
    proba = clf.predict_proba(X_pred)[0]
    return label, dict(zip(clf.classes_, proba))


# ------------------------
# Load historical data
# ------------------------
myor = read_csv_if_exists("myor.csv")
if myor is not None:
    myor.columns = [c.strip() for c in myor.columns]
    if str(myor.columns[0]).strip().lower() != "date":
        myor = myor.rename(columns={myor.columns[0]: "date"})
    myor_col = next((c for c in myor.columns if c.lower() in ["myor","reference_rate","rate"]), None)
    if myor_col and myor_col != "myor":
        myor = myor.rename(columns={myor_col: "myor"})
    vol_col = next((c for c in myor.columns if "volume" in c.lower() or "aggregate" in c.lower()), None)
    if vol_col and vol_col != "aggregate_volume":
        myor = myor.rename(columns={vol_col: "aggregate_volume"})
    myor["date"] = pd.to_datetime(myor["date"], errors="coerce").dt.date
    if "myor" in myor.columns:
        myor["myor"] = pd.to_numeric(myor["myor"], errors="coerce")
    if "aggregate_volume" in myor.columns:
        myor["aggregate_volume"] = pd.to_numeric(myor["aggregate_volume"], errors="coerce")
    myor = myor.dropna(subset=["date"]).reset_index(drop=True)

interbank = read_csv_if_exists("interbank_rates.csv")
if interbank is not None:
    interbank.columns = [c.strip() for c in interbank.columns]
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
    interbank_vol.columns = [c.strip() for c in interbank_vol.columns]
    if str(interbank_vol.columns[0]).strip().lower() != "date":
        interbank_vol = interbank_vol.rename(columns={interbank_vol.columns[0]: "date"})
    interbank_vol["date"] = pd.to_datetime(interbank_vol["date"], errors="coerce").dt.date
    if "volume" in interbank_vol.columns:
        interbank_vol["volume"] = pd.to_numeric(interbank_vol["volume"], errors="coerce")
    interbank_vol = interbank_vol.dropna(subset=["date"]).reset_index(drop=True)

# ------------------------
# Feature generator
# ------------------------
def generate_features(pred_date: date, lookback_days=7):
    feat = {"date": pred_date}

    # MYOR features
    if myor is not None:
        lb_start = pred_date - timedelta(days=lookback_days)
        window = myor[(myor["date"] > lb_start) & (myor["date"] <= pred_date)]
        if not window.empty and "myor" in window.columns:
            feat["myor_mean_7d"] = window["myor"].mean()
            feat["myor_last"] = window["myor"].iloc[-1]
        else:
            feat["myor_mean_7d"] = np.nan
            feat["myor_last"] = np.nan
    else:
        feat["myor_mean_7d"] = np.nan
        feat["myor_last"] = np.nan

    # Interbank overnight
    if interbank is not None and "tenor" in interbank.columns:
        lb_start = pred_date - timedelta(days=lookback_days)
        ov = interbank[(interbank["tenor"].str.lower().str.contains("overnight", na=False)) &
                       (interbank["date"] > lb_start) & (interbank["date"] <= pred_date)]
        feat["overnight_mean_7d"] = ov["rate"].mean() if not ov.empty else np.nan
        feat["overnight_last"] = ov["rate"].iloc[-1] if not ov.empty else np.nan
    else:
        feat["overnight_mean_7d"] = np.nan
        feat["overnight_last"] = np.nan

    # Interbank 1-month
    if interbank is not None and "tenor" in interbank.columns:
        m1 = interbank[(interbank["tenor"].str.contains("1_month|1 month|1_month", case=False, regex=True)) &
                       (interbank["date"] > pred_date - timedelta(days=lookback_days)) & (interbank["date"] <= pred_date)]
        feat["m1_mean_7d"] = m1["rate"].mean() if not m1.empty else np.nan
    else:
        feat["m1_mean_7d"] = np.nan

    # Volume features
    if interbank_vol is not None and "volume" in interbank_vol.columns:
        lb_start = pred_date - timedelta(days=lookback_days)
        vv = interbank_vol[(interbank_vol["date"] > lb_start) & (interbank_vol["date"] <= pred_date)]
        feat["vol_mean_7d"] = vv["volume"].mean() if not vv.empty else np.nan
        feat["vol_sum_7d"] = vv["volume"].sum() if not vv.empty else np.nan
    else:
        feat["vol_mean_7d"] = np.nan
        feat["vol_sum_7d"] = np.nan

    # Spread
    if feat["myor_last"] is not np.nan:
        feat["myor_minus_opr"] = np.nan  # 因为预测时没有当前 OPR，暂时置空
    else:
        feat["myor_minus_opr"] = np.nan

    # Diff features
    feat["myor_diff"] = feat["myor_last"] - feat["myor_mean_7d"] if pd.notna(feat["myor_last"]) else 0.0
    feat["overnight_diff"] = feat["overnight_last"] - feat["overnight_mean_7d"] if pd.notna(feat["overnight_last"]) else 0.0


    # 构建 DataFrame
    X_pred = pd.DataFrame([feat])
    for f in features:
        if f not in X_pred.columns or pd.isna(X_pred.at[0, f]):
            X_pred[f] = 0.0  # 没有数据就用0（训练时SimpleImputer策略是mean, 对新数据0也安全）

    return X_pred[features]

# ------------------------
# Predict
# ------------------------
def predict_opr(pred_date: str):
    pred_date_obj = datetime.strptime(pred_date, "%Y-%m-%d").date()
    X_pred = generate_features(pred_date_obj)
    label = clf.predict(X_pred)[0]
    proba = clf.predict_proba(X_pred)[0]
    return label, dict(zip(clf.classes_, proba))

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    print("[info] Predicting for next OPR decision dates...")

    if OPR_DECISIONS:  # 确保还有未来的日期
        fd = OPR_DECISIONS[0]  # 只取最近的那个日期
        label, proba = predict_opr(fd)
        proba = {k: float(v) for k, v in proba.items()}  # 转 float 方便展示
        print(f"Date: {fd}, Predicted OPR: {label}, Probabilities: {proba}")
    else:
        print("[warning] No future OPR decision dates found.")