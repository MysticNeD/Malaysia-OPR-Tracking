# models/visualize.py
"""
Visualize data and model predictions for Malaysia OPR tracker.

Usage examples:
    python visualize.py --future-dates 2025-09-05 2025-11-06
    python visualize.py --from-schedule   # read data/opr_schedule.txt (one date per line)
    python visualize.py --save-csv        # save predictions to data/predictions.csv

Outputs:
  - outputs/oprs_history.png
  - outputs/myor_series.png
  - outputs/interbank_overnight_m1.png
  - outputs/pred_probs_<YYYY-MM-DD>.png
  - data/predictions.csv (if --save-csv)
"""
from pathlib import Path
from datetime import datetime, timedelta, date
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import predict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (ROOT / "data").resolve()
MODEL_DIR = (ROOT / "models").resolve()
OUT_DIR = (ROOT / "outputs").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Helpful defaults mirroring train/predict code
LOOKBACK_DAYS = 7
MYOR_START_DATE = date(2022, 1, 1)  # only use MYOR for dates >= this (same as train.py)

# ------------------------
# utils
# ------------------------
def safe_read_csv(path: Path, **kwargs):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=str, **kwargs)
    except Exception as e:
        print(f"[warn] failed to read {path}: {e}", file=sys.stderr)
        return None

def parse_date_col(df: pd.DataFrame, col='date'):
    df = df.copy()
    if col not in df.columns:
        return df
    df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
    return df

# ------------------------
# load model & features
# ------------------------
def load_model_bundle():
    p = MODEL_DIR / "model.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    mb = joblib.load(p)
    model = mb.get("model")
    features = mb.get("features")
    if features is None:
        raise RuntimeError("Model bundle missing 'features' list")
    return model, features

# ------------------------
# load datasets
# ------------------------
def load_datasets():
    opr = safe_read_csv(DATA_DIR / "oprs.csv")
    if opr is not None:
        opr.columns = [c.strip() for c in opr.columns]
        if not any(c.lower() == "date" for c in opr.columns):
            opr = opr.rename(columns={opr.columns[0]: "date"})
        opr = parse_date_col(opr, "date")
        # try to map rate -> new_opr_level
        lower = [c.lower() for c in opr.columns]
        if "new_opr_level" not in lower and "rate" in lower:
            real = [c for c in opr.columns if c.lower() == "rate"][0]
            opr = opr.rename(columns={real: "new_opr_level"})
        if "new_opr_level" in opr.columns:
            opr["new_opr_level"] = pd.to_numeric(opr["new_opr_level"], errors="coerce")
        opr = opr.sort_values("date").reset_index(drop=True)

    myor = safe_read_csv(DATA_DIR / "myor.csv")
    if myor is not None:
        myor.columns = [c.strip() for c in myor.columns]
        if not any(c.lower() == "date" for c in myor.columns):
            myor = myor.rename(columns={myor.columns[0]: "date"})
        myor = parse_date_col(myor, "date")
        # map possible myor column
        lower = [c.lower() for c in myor.columns]
        candidate = None
        for cand in ["myor", "reference_rate", "rate"]:
            if cand in lower:
                candidate = myor.columns[lower.index(cand)]
                break
        if candidate and candidate != "myor":
            myor = myor.rename(columns={candidate: "myor"})
        # volume column
        vol_candidates = [c for c in myor.columns if "volume" in c.lower() or "aggregate" in c.lower()]
        if vol_candidates and vol_candidates[0] != "aggregate_volume":
            myor = myor.rename(columns={vol_candidates[0]: "aggregate_volume"})
        if "myor" in myor.columns:
            myor["myor"] = pd.to_numeric(myor["myor"], errors="coerce")
        if "aggregate_volume" in myor.columns:
            myor["aggregate_volume"] = pd.to_numeric(myor["aggregate_volume"], errors="coerce")
        myor = myor.dropna(subset=["date"]).reset_index(drop=True)

    interbank = safe_read_csv(DATA_DIR / "interbank_rates.csv")
    if interbank is not None:
        interbank.columns = [c.strip() for c in interbank.columns]
        if not any(c.lower() == "date" for c in interbank.columns):
            interbank = interbank.rename(columns={interbank.columns[0]: "date"})
        interbank = parse_date_col(interbank, "date")
        # normalize names
        lower = [c.lower() for c in interbank.columns]
        if "rate" in lower:
            rate_col = [c for c in interbank.columns if c.lower() == "rate"][0]
            if rate_col != "rate":
                interbank = interbank.rename(columns={rate_col: "rate"})
            interbank["rate"] = pd.to_numeric(interbank["rate"], errors="coerce")
        if "tenor" in lower:
            tenor_col = [c for c in interbank.columns if c.lower() == "tenor"][0]
            if tenor_col != "tenor":
                interbank = interbank.rename(columns={tenor_col: "tenor"})
            interbank["tenor"] = interbank["tenor"].astype(str)
        interbank = interbank.dropna(subset=["date"]).reset_index(drop=True)

    interbank_vol = safe_read_csv(DATA_DIR / "interbank_volumes.csv")
    if interbank_vol is not None:
        interbank_vol.columns = [c.strip() for c in interbank_vol.columns]
        if not any(c.lower() == "date" for c in interbank_vol.columns):
            interbank_vol = interbank_vol.rename(columns={interbank_vol.columns[0]: "date"})
        interbank_vol = parse_date_col(interbank_vol, "date")
        vol_col = next((c for c in interbank_vol.columns if "vol" in c.lower() or "volume" in c.lower()), None)
        if vol_col and vol_col != "volume":
            interbank_vol = interbank_vol.rename(columns={vol_col: "volume"})
        if "volume" in interbank_vol.columns:
            interbank_vol["volume"] = pd.to_numeric(interbank_vol["volume"], errors="coerce")
        interbank_vol = interbank_vol.dropna(subset=["date"]).reset_index(drop=True)

    fast = safe_read_csv(DATA_DIR / "fast.csv")
    if fast is not None:
        fast.columns = [c.strip() for c in fast.columns]
        # try parse date-like col
        date_cols = [c for c in fast.columns if "date" in c.lower() or "as_at" in c.lower()]
        if date_cols:
            fast = fast.rename(columns={date_cols[0]: "date"})
            fast = parse_date_col(fast, "date")

    return opr, myor, interbank, interbank_vol, fast

# ------------------------
# feature gen (mirror predict.py / train.py logic)
# ------------------------
def generate_features_for_model(pred_date: date, features_list, opr_df, myor_df, interbank_df, interbank_vol_df, lookback_days=LOOKBACK_DAYS):
    feat = {"date": pred_date}
    lb_start = pred_date - timedelta(days=lookback_days)

    # myor usage only if pred_date >= MYOR_START_DATE and myor exists
    if myor_df is not None and pred_date >= MYOR_START_DATE:
        window = myor_df[(myor_df["date"] > lb_start) & (myor_df["date"] <= pred_date)]
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
    if interbank_df is not None and "tenor" in interbank_df.columns and "rate" in interbank_df.columns:
        ov = interbank_df[
            (interbank_df["tenor"].str.lower().str.contains("overnight", na=False)) &
            (interbank_df["date"] > lb_start) & (interbank_df["date"] <= pred_date)
        ]
        feat["overnight_mean_7d"] = ov["rate"].mean() if not ov.empty else np.nan
        feat["overnight_last"] = ov["rate"].iloc[-1] if not ov.empty else np.nan
    else:
        feat["overnight_mean_7d"] = np.nan
        feat["overnight_last"] = np.nan

    # Interbank 1-month
    if interbank_df is not None and "tenor" in interbank_df.columns and "rate" in interbank_df.columns:
        m1 = interbank_df[
            interbank_df["tenor"].str.contains("1_month|1 month|1_month", case=False, regex=True) &
            (interbank_df["date"] > lb_start) & (interbank_df["date"] <= pred_date)
        ]
        feat["m1_mean_7d"] = m1["rate"].mean() if not m1.empty else np.nan
    else:
        feat["m1_mean_7d"] = np.nan

    # Volume features: use overall volume (not tenor-specific) as train.py did
    if interbank_vol_df is not None and "volume" in interbank_vol_df.columns:
        vv = interbank_vol_df[(interbank_vol_df["date"] > lb_start) & (interbank_vol_df["date"] <= pred_date)]
        feat["vol_mean_7d"] = vv["volume"].mean() if not vv.empty else np.nan
        feat["vol_sum_7d"] = vv["volume"].sum() if not vv.empty else np.nan
    else:
        feat["vol_mean_7d"] = np.nan
        feat["vol_sum_7d"] = np.nan

    # myor_minus_opr: need the latest OPR at or before pred_date
    if opr_df is not None and "new_opr_level" in opr_df.columns:
        prev_opr = opr_df[opr_df["date"] <= pred_date].sort_values("date")
        if not prev_opr.empty:
            latest_opr = prev_opr["new_opr_level"].iloc[-1]
            if pd.notna(feat["myor_last"]) and pd.notna(latest_opr):
                feat["myor_minus_opr"] = feat["myor_last"] - float(latest_opr)
            else:
                feat["myor_minus_opr"] = np.nan
        else:
            feat["myor_minus_opr"] = np.nan
    else:
        feat["myor_minus_opr"] = np.nan

    # Diff features
    feat["myor_diff"] = (feat["myor_last"] - feat["myor_mean_7d"]) if pd.notna(feat["myor_last"]) and pd.notna(feat["myor_mean_7d"]) else 0.0
    feat["overnight_diff"] = (feat["overnight_last"] - feat["overnight_mean_7d"]) if pd.notna(feat["overnight_last"]) and pd.notna(feat["overnight_mean_7d"]) else 0.0

    # Build DataFrame that includes exactly the features model expects
    X = pd.DataFrame([feat])
    # ensure all required features exist
    for f in features_list:
        if f not in X.columns:
            X[f] = 0.0
        else:
            # replace NA with 0.0 (consistent with predict.py's default fallback)
            if pd.isna(X.at[0, f]):
                X.at[0, f] = 0.0
    # keep only model features and ensure numeric
    X = X[features_list].astype(float)
    return X

# ------------------------
# plotting functions (each plot in separate figure)
# ------------------------
def plot_opr_history(opr_df: pd.DataFrame, out_path: Path):
    if opr_df is None or opr_df.empty:
        print("[info] no opr data to plot")
        return
    # step plot for policy rates over time
    df = opr_df.sort_values("date")
    dates = pd.to_datetime(df["date"])
    rates = df["new_opr_level"].astype(float)
    plt.figure(figsize=(10, 4))
    plt.step(dates, rates, where="post")
    plt.scatter(dates, rates, s=30)
    plt.title("OPR history (policy rate)")
    plt.xlabel("Date")
    plt.ylabel("OPR (%)")
    plt.grid(True, linestyle='--', linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[ok] saved {out_path}")

def plot_myor_series(myor_df: pd.DataFrame, out_path: Path):
    if myor_df is None or "myor" not in myor_df.columns or myor_df.empty:
        print("[info] no MYOR data to plot")
        return
    df = myor_df.sort_values("date")
    dates = pd.to_datetime(df["date"])
    plt.figure(figsize=(10, 4))
    plt.plot(dates, df["myor"].astype(float))
    plt.title("MYOR series")
    plt.xlabel("Date")
    plt.ylabel("MYOR (%)")
    plt.grid(True, linestyle='--', linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[ok] saved {out_path}")

def plot_interbank_overnight_and_m1(interbank_df: pd.DataFrame, out_path: Path):
    if interbank_df is None or interbank_df.empty:
        print("[info] no interbank rates data to plot")
        return
    df = interbank_df.copy()
    # pick overnight and 1_month
    overnight = df[df["tenor"].str.lower().str.contains("overnight", na=False)]
    m1 = df[df["tenor"].str.contains("1_month|1 month|1_month", case=False, regex=True)]
    plt.figure(figsize=(10, 4))
    if not overnight.empty:
        d_ov = pd.to_datetime(overnight["date"])
        plt.plot(d_ov, overnight["rate"].astype(float), label="overnight")
    if not m1.empty:
        d_m1 = pd.to_datetime(m1["date"])
        plt.plot(d_m1, m1["rate"].astype(float), label="1_month")
    plt.title("Interbank rates: overnight & 1-month")
    plt.xlabel("Date")
    plt.ylabel("Rate (%)")
    plt.grid(True, linestyle='--', linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[ok] saved {out_path}")

def plot_prediction_bars(pred_date: str, probs: dict, out_path: Path):
    labels = list(probs.keys())
    vals = [float(probs[k]) for k in labels]
    plt.figure(figsize=(6, 3.5))
    plt.bar(labels, vals)
    plt.ylim(0, 1.0)
    plt.title(f"Predicted probs for {pred_date}")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[ok] saved {out_path}")

# ------------------------
# main
# ------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(prog="visualize.py", description="Visualize OPR data and model predictions")
    parser.add_argument("--future-dates", nargs="+", help="List of YYYY-MM-DD future decision dates to predict")
    parser.add_argument("--from-schedule", action="store_true", help="Read data/opr_schedule.txt for future dates (one per line)")
    parser.add_argument("--save-csv", action="store_true", help="Save predictions to data/predictions.csv")
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS, help="Lookback window in days for features")
    args = parser.parse_args(argv)

    opr, myor, interbank, interbank_vol, fast = load_datasets()

    # load model
    try:
        clf, features = load_model_bundle()
    except Exception as e:
        print(f"[error] failed to load model: {e}", file=sys.stderr)
        return

    # choose future dates
    future_dates = ["2025-09-04", "2025-11-06"]
    if args.future_dates:
        future_dates = args.future_dates
    elif args.from_schedule:
        sched = DATA_DIR / "opr_schedule.txt"
        if sched.exists():
            with open(sched, "r", encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            # only future lines
            today = datetime.now().date()
            for ln in lines:
                try:
                    d = datetime.strptime(ln, "%Y-%m-%d").date()
                    if d >= today:
                        future_dates.append(ln)
                except Exception:
                    continue
        else:
            print("[warn] data/opr_schedule.txt not found; pass --future-dates or create the file.", file=sys.stderr)
    else:
        print("[info] No future dates provided. Use --future-dates or create data/opr_schedule.txt. Exiting.")
        # still generate historical plots
    # generate plots (historical)
    plot_opr_history(opr, OUT_DIR / "oprs_history.png")
    plot_myor_series(myor, OUT_DIR / "myor_series.png")
    plot_interbank_overnight_and_m1(interbank, OUT_DIR / "interbank_overnight_m1.png")

    # predict for each future date provided
    predictions = []
    for fd in future_dates:
        try:
            pred_date_obj = datetime.strptime(fd, "%Y-%m-%d").date()
        except Exception:
            print(f"[warn] invalid date format: {fd}, skipping", file=sys.stderr)
            continue
        Xpred = generate_features_for_model(pred_date_obj, features, opr, myor, interbank, interbank_vol, lookback_days=args.lookback_days)
        # model may be CalibratedClassifierCV or pipeline; ensure predict_proba exists
        try:
            probs_arr = clf.predict_proba(Xpred)[0]
            classes = list(clf.classes_)
            probs = dict(zip(classes, probs_arr))
            # ensure consistent ordering of keys 'down','same','up' if possible
            # Save CSV row
            pred_label = clf.predict(Xpred)[0]
        except Exception as e:
            print(f"[error] model prediction failed for {fd}: {e}", file=sys.stderr)
            probs = {c: 0.0 for c in ["down", "same", "up"]}
            pred_label = None

        # normalize probs keys to include down/same/up
        normalized = {k: float(probs.get(k, 0.0)) for k in ["down", "same", "up"]}
        predictions.append({
            "date": fd,
            "pred_label": pred_label,
            "prob_down": normalized["down"],
            "prob_same": normalized["same"],
            "prob_up": normalized["up"]
        })
        # plot bar
        plot_prediction_bars(fd, normalized, OUT_DIR / f"pred_probs_{fd}.png")
        print(f"[pred] {fd} -> {pred_label} probs={normalized}")

    # optionally save predictions CSV
    if args.save_csv and predictions:
        out_df = pd.DataFrame(predictions)
        out_path = DATA_DIR / "predictions.csv"
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[ok] Saved predictions CSV: {out_path}")

    print("[done] visualization complete. Outputs in ./outputs/")

if __name__ == "__main__":
    main()
