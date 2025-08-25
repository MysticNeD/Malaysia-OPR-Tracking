# models/train.py
"""
Train a multi-class classifier to predict probability of next OPR movement (up/same/down)
based on today's data and historical features.
"""
import os
from pathlib import Path
from datetime import timedelta, date, datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (ROOT / "data").resolve()
MODEL_DIR = (ROOT / "models").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Helper: read CSV if exists
# -------------------------------
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

# -------------------------------
# Prepare dataset for training
# -------------------------------
def prepare_dataset():
    # Load OPR history
    opr = read_csv_if_exists("oprs.csv")
    if opr is None:
        raise SystemExit("data/oprs.csv missing.")
    opr["date"] = pd.to_datetime(opr["date"], errors="coerce").dt.date
    opr["new_opr_level"] = pd.to_numeric(opr["new_opr_level"], errors="coerce")
    opr = opr.dropna(subset=["date", "new_opr_level"]).sort_values("date").reset_index(drop=True)
    
    # Build label based on next OPR
    opr["next_new_opr"] = opr["new_opr_level"].shift(-1)
    opr = opr.dropna(subset=["next_new_opr"]).reset_index(drop=True)
    
    def label_row(row):
        cur = float(row["new_opr_level"])
        nxt = float(row["next_new_opr"])
        if nxt > cur + 1e-8:
            return "up"
        elif nxt < cur - 1e-8:
            return "down"
        else:
            return "same"
    opr["label"] = opr.apply(label_row, axis=1)
    
    # Load MYOR (2022+)
    myor = read_csv_if_exists("myor.csv")
    if myor is not None:
        myor.columns = [c.strip() for c in myor.columns]
        if str(myor.columns[0]).strip().lower() != "date":
            myor = myor.rename(columns={myor.columns[0]: "date"})
        myor["date"] = pd.to_datetime(myor["date"], errors="coerce").dt.date
        myor["myor"] = pd.to_numeric(myor.get("myor", myor.columns[1]), errors="coerce")
        myor["aggregate_volume"] = pd.to_numeric(myor.get("aggregate_volume", 0), errors="coerce")
        myor = myor.dropna(subset=["date"]).reset_index(drop=True)
    
    # Load interbank rates
    interbank = read_csv_if_exists("interbank_rates.csv")
    if interbank is not None:
        interbank.columns = [c.strip() for c in interbank.columns]
        if str(interbank.columns[0]).strip().lower() != "date":
            interbank = interbank.rename(columns={interbank.columns[0]: "date"})
        interbank["date"] = pd.to_datetime(interbank["date"], errors="coerce").dt.date
        interbank["rate"] = pd.to_numeric(interbank.get("rate", 0), errors="coerce")
        interbank["tenor"] = interbank.get("tenor", "").astype(str)
        interbank = interbank.dropna(subset=["date"]).reset_index(drop=True)
    
    # Load interbank volumes
    interbank_vol = read_csv_if_exists("interbank_volumes.csv")
    if interbank_vol is not None:
        interbank_vol.columns = [c.strip() for c in interbank_vol.columns]
        if str(interbank_vol.columns[0]).strip().lower() != "date":
            interbank_vol = interbank_vol.rename(columns={interbank_vol.columns[0]: "date"})
        interbank_vol["date"] = pd.to_datetime(interbank_vol["date"], errors="coerce").dt.date
        interbank_vol["volume"] = pd.to_numeric(interbank_vol.get("volume", 0), errors="coerce")
        interbank_vol = interbank_vol.dropna(subset=["date"]).reset_index(drop=True)
    
    # -------------------------------
    # Generate features for each OPR
    # -------------------------------
    rows = []
    for idx, r in opr.iterrows():
        ann_date = r["date"]
        next_ann_date = opr["date"].iloc[idx + 1] if idx + 1 < len(opr) else ann_date + timedelta(days=60)
        lookback_days = (next_ann_date - ann_date).days
        lookback_days = max(lookback_days, 60)  # fallback
        
        feat = {"date": ann_date, "new_opr_level": r["new_opr_level"], "label": r["label"]}
        
        # MYOR features
        if myor is not None:
            myor_window = myor[(myor["date"] > ann_date - timedelta(days=lookback_days)) & (myor["date"] <= ann_date)]
            feat["myor_mean"] = myor_window["myor"].mean() if not myor_window.empty else np.nan
            feat["myor_last"] = myor_window["myor"].iloc[-1] if not myor_window.empty else np.nan
            feat["myor_vol_mean"] = myor_window["aggregate_volume"].mean() if not myor_window.empty else np.nan
            feat["myor_vol_last"] = myor_window["aggregate_volume"].iloc[-1] if not myor_window.empty else np.nan
        else:
            feat["myor_mean"] = np.nan
            feat["myor_last"] = np.nan
            feat["myor_vol_mean"] = np.nan
            feat["myor_vol_last"] = np.nan
        
        # Interbank overnight
        if interbank is not None:
            ov_window = interbank[(interbank["tenor"].str.lower().str.contains("overnight", na=False)) &
                                  (interbank["date"] > ann_date - timedelta(days=lookback_days)) &
                                  (interbank["date"] <= ann_date)]
            feat["overnight_mean"] = ov_window["rate"].mean() if not ov_window.empty else np.nan
            feat["overnight_last"] = ov_window["rate"].iloc[-1] if not ov_window.empty else np.nan
        else:
            feat["overnight_mean"] = np.nan
            feat["overnight_last"] = np.nan
        
        # Interbank 1-month
        if interbank is not None:
            m1_window = interbank[(interbank["tenor"].str.contains("1_month|1 month", case=False, regex=True)) &
                                  (interbank["date"] > ann_date - timedelta(days=lookback_days)) &
                                  (interbank["date"] <= ann_date)]
            feat["m1_mean"] = m1_window["rate"].mean() if not m1_window.empty else np.nan
        else:
            feat["m1_mean"] = np.nan
        
        # Interbank volume
        if interbank_vol is not None:
            vol_window = interbank_vol[(interbank_vol["date"] > ann_date - timedelta(days=lookback_days)) &
                                       (interbank_vol["date"] <= ann_date)]
            feat["vol_mean"] = vol_window["volume"].mean() if not vol_window.empty else np.nan
            feat["vol_sum"] = vol_window["volume"].sum() if not vol_window.empty else np.nan
        else:
            feat["vol_mean"] = np.nan
            feat["vol_sum"] = np.nan
        
        # Spread & diff features
        try:
            feat["myor_minus_opr"] = feat["myor_last"] - float(r["new_opr_level"]) if pd.notna(feat["myor_last"]) else np.nan
        except:
            feat["myor_minus_opr"] = np.nan
        
        feat["myor_diff"] = feat["myor_last"] - feat["myor_mean"] if pd.notna(feat["myor_last"]) else 0.0
        feat["overnight_diff"] = feat["overnight_last"] - feat["overnight_mean"] if pd.notna(feat["overnight_last"]) else 0.0
        
        rows.append(feat)
    
    Xdf = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    
    features = ["myor_mean", "myor_last", "overnight_mean", "overnight_last",
                "m1_mean", "vol_mean", "vol_sum", "myor_minus_opr",
                "myor_vol_mean", "myor_vol_last", "myor_diff", "overnight_diff"]
    
    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    Xdf[features] = pd.DataFrame(imputer.fit_transform(Xdf[features]), columns=features)
    
    y = Xdf["label"]
    X = Xdf[features]
    
    # Small sample augmentation
    if len(Xdf) < 100:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
    else:
        X_res, y_res = X, y
    
    # Train/test split
    stratify_arg = y_res if y_res.nunique() > 1 and y_res.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=stratify_arg)
    
    # Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(multi_class="multinomial", max_iter=2000, class_weight="balanced", solver="lbfgs"))
    ])
    
    clf = pipe
    clf.fit(X_train, y_train)
    
    # Save model
    model_bundle = {"model": clf, "features": features}
    joblib.dump(model_bundle, MODEL_DIR / "model.pkl")
    print(f"[ok] model saved to {MODEL_DIR / 'model.pkl'}")
    
    # Evaluate
    ypred = clf.predict(X_test)
    acc = accuracy_score(y_test, ypred)
    cr = classification_report(y_test, ypred)
    
    with open(MODEL_DIR / "metrics.txt", "w", encoding="utf-8") as fh:
        fh.write(f"accuracy: {acc}\n\n")
        fh.write(cr)
    
    print("Train (in-sample) score:", clf.score(X_train, y_train))
    print("Reported accuracy (eval):", acc)
    print("Classification report (eval):\n", cr)
    
    return model_bundle, Xdf

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    model_bundle, prepared = prepare_dataset()
    print("Done. Prepared samples:", len(prepared))
