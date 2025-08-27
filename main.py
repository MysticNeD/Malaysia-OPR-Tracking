# main.py
"""
Run full pipeline for myopr-watch:
1. Fetch latest BNM data
2. Train model
3. Predict next OPR movement based on today's data
4. Save results
"""

from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import json
import sys

# ------------------ 配置路径 ------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = (ROOT / "data").resolve()
MODEL_DIR = (ROOT / "models").resolve()
OUTPUT_DIR = (ROOT / "outputs").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ 导入模块 ------------------
from scripts.fetch_bnm import main as fetch_bnm_main
from models.train import prepare_dataset
from models.predict import predict_opr, generate_features, OPR_DECISIONS, _load_model

# ------------------ 主流程 ------------------
def main():
    print("[info] Fetching latest BNM data...")
    try:
        fetch_bnm_main()
    except Exception as e:
        print("[error] fetch_bnm failed:", e, file=sys.stderr)

    print("[info] Training model...")
    try:
        model_bundle, prepared = prepare_dataset()
    except Exception as e:
        print("[error] training failed:", e, file=sys.stderr)
        return

    clf, _features = _load_model()

    print("[info] Predicting next OPR decision probabilities...")
    today = datetime.now().date()

    # 获取下一个 OPR 日期
    upcoming_dates = [d for d in OPR_DECISIONS if datetime.strptime(d, "%Y-%m-%d").date() >= today]
    if not upcoming_dates:
        print("[warning] No upcoming OPR decisions")
        return

    # 获取上一次 OPR 日期
    past_oprs = [d for d in OPR_DECISIONS if datetime.strptime(d, "%Y-%m-%d").date() < today]
    last_opr_date = datetime.strptime(max(past_oprs), "%Y-%m-%d").date() if past_oprs else today - timedelta(days=60)
    lookback_days = (today - last_opr_date).days

    predictions = []
    for fd in upcoming_dates:
        try:
            # 用今天的数据生成特征，window 从 last OPR 到今天
            X_pred = generate_features(pred_date=today, lookback_days=lookback_days)
            label = clf.predict(X_pred)[0]
            proba_values = clf.predict_proba(X_pred)[0]
            proba = dict(zip(clf.classes_, proba_values))

            predictions.append({
                "date": fd,
                "predicted_opr": label,
                **proba
            })

            print(f"Next OPR date: {fd}, Predicted OPR: {label}, Probabilities: {proba}")
        except Exception as e:
            print(f"[error] failed to predict for {fd}: {e}", file=sys.stderr)

    # 保存 CSV
    if predictions:
        out_csv = DATA_DIR / "predictions.csv"
        pd.DataFrame(predictions).to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[info] Saved predictions CSV: {out_csv}")

        # 保存 JSON
        out_json = DATA_DIR / "predictions.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"[info] Saved predictions JSON: {out_json}")


if __name__ == "__main__":
    main()
