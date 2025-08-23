# models/train.py
"""
Train a simple multi-class classifier to predict next OPR decision (up/same/down)
Based on: data/oprs.csv (announcement dates), data/myor.csv (daily), data/interbank_rates.csv (long form: date, tenor, rate),
and optionally data/interbank_volumes.csv.

Output:
 - models/model.pkl  (joblib dump of {"model": clf, "features": features})
 - models/metrics.txt (train/test scores + classification report)
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
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (ROOT / "data").resolve()
MODEL_DIR = (ROOT / "models").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Helper: read CSV if exists (robust: don't force parse_dates here)
def read_csv_if_exists(name):
    p = DATA_DIR / name
    if not p.exists():
        print(f"[warn] {p} not found")
        return None
    try:
        df = pd.read_csv(p, dtype=str)  # read everything as string for robust cleaning
        # strip column names
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        return df
    except Exception as e:
        print(f"[warn] failed to read {p}: {e}")
        return None

def prepare_dataset():
    # 1) Load opr announcements (these are the policy decision dates)
    opr = read_csv_if_exists("oprs.csv")
    if opr is None:
        raise SystemExit("data/oprs.csv missing. Run fetch/normalize first.")

    # Normalize columns
    cols_lower = [c.lower() for c in opr.columns]
    if "rate" in cols_lower and "new_opr_level" not in cols_lower:
        real_rate = [c for c in opr.columns if c.lower() == "rate"][0]
        opr = opr.rename(columns={real_rate: "new_opr_level"})

    # ensure date col exists
    if not any(c.lower() == "date" for c in opr.columns):
        opr = opr.rename(columns={opr.columns[0]: "date"})
    opr["date"] = pd.to_datetime(opr["date"], errors="coerce").dt.date
    opr = opr.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ensure new_opr_level col exists
    if not any(c.lower() == "new_opr_level" for c in opr.columns):
        raise SystemExit("oprs.csv missing 'new_opr_level' column.")

    opr["new_opr_level"] = pd.to_numeric(opr["new_opr_level"], errors="coerce")

    # map change_in_opr if exists
    if any(c.lower() == "change_in_opr" for c in opr.columns):
        real_change = [c for c in opr.columns if c.lower() == "change_in_opr"][0]
        opr = opr.rename(columns={real_change: "change_in_opr"})
        opr["change_in_opr"] = pd.to_numeric(opr["change_in_opr"], errors="coerce")
    else:
        opr["change_in_opr"] = np.nan

    # build labels based on next announcement's new_opr_level
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

    # 2) Load daily MYOR and interbank_rates
    myor = read_csv_if_exists("myor.csv")
    if myor is not None:
        myor.columns = [c.strip() if isinstance(c, str) else c for c in myor.columns]
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
        interbank.columns = [c.strip() if isinstance(c, str) else c for c in interbank.columns]
        if str(interbank.columns[0]).strip().lower() != "date":
            interbank = interbank.rename(columns={interbank.columns[0]: "date"})
        interbank["date"] = pd.to_datetime(interbank["date"], errors="coerce").dt.date
        rate_col = next((c for c in interbank.columns if c.lower() == "rate"), None)
        if rate_col:
            interbank = interbank.rename(columns={rate_col: "rate"})
            interbank["rate"] = pd.to_numeric(interbank["rate"], errors="coerce")
        tenor_col = next((c for c in interbank.columns if c.lower() == "tenor"), None)
        if tenor_col and tenor_col != "tenor":
            interbank = interbank.rename(columns={tenor_col: "tenor"})
        if "tenor" in interbank.columns:
            interbank["tenor"] = interbank["tenor"].astype(str)
        interbank = interbank.dropna(subset=["date"]).reset_index(drop=True)

    interbank_vol = read_csv_if_exists("interbank_volumes.csv")
    if interbank_vol is not None:
        interbank_vol.columns = [c.strip() if isinstance(c, str) else c for c in interbank_vol.columns]
        if str(interbank_vol.columns[0]).strip().lower() != "date":
            interbank_vol = interbank_vol.rename(columns={interbank_vol.columns[0]: "date"})
        interbank_vol["date"] = pd.to_datetime(interbank_vol["date"], errors="coerce").dt.date
        vol_col = next((c for c in interbank_vol.columns if "vol" in c.lower() or "volume" in c.lower()), None)
        if vol_col and vol_col != "volume":
            interbank_vol = interbank_vol.rename(columns={vol_col: "volume"})
        if "volume" in interbank_vol.columns:
            interbank_vol["volume"] = pd.to_numeric(interbank_vol["volume"], errors="coerce")
        interbank_vol = interbank_vol.dropna(subset=["date"]).reset_index(drop=True)

    # 3) For each announcement date, compute features using lookback window
    rows = []
    lookback_days = 7
    myor_start_date = date(2022, 1, 1)
    for idx, r in opr.iterrows():
        ann_date = r["date"]
        feat = {"date": ann_date, "new_opr_level": r.get("new_opr_level", np.nan), "label": r["label"]}

        # MYOR features (only 2022+)
        if myor is not None and ann_date >= myor_start_date:
            lb_start = ann_date - timedelta(days=lookback_days)
            window = myor[(myor["date"] > lb_start) & (myor["date"] <= ann_date)]
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
        if interbank is not None and "tenor" in interbank.columns and "rate" in interbank.columns:
            lb_start = ann_date - timedelta(days=lookback_days)
            ov = interbank[(interbank["tenor"].str.lower().str.contains("overnight", na=False)) & 
                           (interbank["date"] > lb_start) & (interbank["date"] <= ann_date)]
            feat["overnight_mean_7d"] = ov["rate"].mean() if not ov.empty else np.nan
            feat["overnight_last"] = ov["rate"].iloc[-1] if not ov.empty else np.nan
        else:
            feat["overnight_mean_7d"] = np.nan
            feat["overnight_last"] = np.nan

        # Interbank 1-month
        if interbank is not None and "tenor" in interbank.columns and "rate" in interbank.columns:
            m1 = interbank[(interbank["tenor"].str.contains("1_month|1 month|1_month", case=False, regex=True)) & 
                           (interbank["date"] > ann_date - timedelta(days=lookback_days)) & (interbank["date"] <= ann_date)]
            feat["m1_mean_7d"] = m1["rate"].mean() if not m1.empty else np.nan
        else:
            feat["m1_mean_7d"] = np.nan

        # Volume features
        if interbank_vol is not None and "volume" in interbank_vol.columns:
            lb_start = ann_date - timedelta(days=lookback_days)
            vv = interbank_vol[(interbank_vol["date"] > lb_start) & (interbank_vol["date"] <= ann_date)]
            feat["vol_mean_7d"] = vv["volume"].mean() if not vv.empty else np.nan
            feat["vol_sum_7d"] = vv["volume"].sum() if not vv.empty else np.nan
        else:
            feat["vol_mean_7d"] = np.nan
            feat["vol_sum_7d"] = np.nan

        # Spread
        try:
            feat["myor_minus_opr"] = (feat["myor_last"] - float(r.get("new_opr_level"))) if (not pd.isna(feat["myor_last"]) and pd.notna(r.get("new_opr_level"))) else np.nan
        except Exception:
            feat["myor_minus_opr"] = np.nan
        rows.append(feat)

    Xdf = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    Xdf = Xdf.dropna(subset=["label"]).reset_index(drop=True)

    # features list
    features = ["myor_mean_7d", "myor_last", "overnight_mean_7d", "overnight_last", 
                "m1_mean_7d", "vol_mean_7d", "vol_sum_7d", "myor_minus_opr"]
    
    # --- 新增差分特征 ---
    Xdf["myor_diff"] = Xdf["myor_last"] - Xdf["myor_mean_7d"]
    Xdf["overnight_diff"] = Xdf["overnight_last"] - Xdf["overnight_mean_7d"]
    features += ["myor_diff", "overnight_diff"]

    # 填充全 NaN 特征为 0
    for f in features:
        if Xdf[f].isna().all():
            Xdf[f] = 0.0

    # ---- Impute missing values ----
    imputer = SimpleImputer(strategy="mean")
    Xdf[features] = Xdf[features].astype(float)
    Xdf[features] = pd.DataFrame(
        imputer.fit_transform(Xdf[features]),
        columns=features,
        index=Xdf.index
    )
    assert not Xdf[features].isna().any().any(), "仍然有NaN，请检查数据处理"

    # Target
    y = Xdf["label"]
    X = Xdf[features].astype(float)

    # 小样本增强
    if len(Xdf) < 100:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
    else:
        X_res, y_res = X, y

    # ---------------------------
    # Train / Split / Calibration
    # ---------------------------
    n_samples = len(X_res)
    class_counts = y_res.value_counts().to_dict()
    min_class_count = min(class_counts.values()) if class_counts else 0
    n_classes = len(class_counts)

    print(f"[info] samples={n_samples}, class_counts={class_counts}")

    stratify_arg = y_res if n_classes > 1 and min_class_count >= 2 else None
    if stratify_arg is not None:
        print("[info] using stratified train/test split")
    else:
        print("[warn] cannot stratify due to small class counts; proceeding without stratify")

    if n_samples < 6:
        print("[warn] very small sample size (<6). Training on all data and skipping calibration if needed.")
        X_train, y_train = X_res, y_res
        X_test, y_test = X_res, y_res
        do_calibration = False
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2, random_state=42, stratify=stratify_arg
            )
        except Exception as e:
            print("[warn] train_test_split failed:", e)
            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2, random_state=42, stratify=None
            )
        do_calibration = True

    # Build pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(multi_class="multinomial", max_iter=2000, class_weight="balanced", solver="lbfgs"))
    ])

    calib_model = None
    if do_calibration and min_class_count >= 2 and n_classes > 1:
        cv_k = min(3, min_class_count)
        if cv_k >= 2:
            try:
                calib_model = CalibratedClassifierCV(pipe, cv=cv_k)
                print(f"[info] using CalibratedClassifierCV with cv={cv_k}")
            except Exception as e:
                print("[warn] failed to init CalibratedClassifierCV:", e)
                calib_model = None

    if calib_model is not None:
        clf = calib_model
        clf.fit(X_train, y_train)
    else:
        clf = pipe
        clf.fit(X_train, y_train)

    # Save model
    model_bundle = {"model": clf, "features": features}
    joblib.dump(model_bundle, MODEL_DIR / "model.pkl")
    print(f"[ok] model saved to {MODEL_DIR / 'model.pkl'}")

    # Evaluate
    try:
        ypred = clf.predict(X_test)
        acc = accuracy_score(y_test, ypred)
        cr = classification_report(y_test, ypred)
    except Exception as e:
        print("[warn] evaluation failed:", e)
        ypred = clf.predict(X_train)
        acc = accuracy_score(y_train, ypred)
        cr = classification_report(y_train, ypred)
        print("[warn] reported metrics are in-sample")

    with open(MODEL_DIR / "metrics.txt", "w", encoding="utf-8") as fh:
        fh.write(f"accuracy: {acc}\n\n")
        fh.write(cr)
    print("Train (in-sample) score:", clf.score(X_train, y_train))
    print("Reported accuracy (eval):", acc)
    print("Classification report (eval):\n", cr)

    return model_bundle, Xdf


if __name__ == "__main__":
    model_bundle, prepared = prepare_dataset()
    print("Done. Prepared samples:", len(prepared))
