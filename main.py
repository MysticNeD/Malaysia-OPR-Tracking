# main.py
"""
Run full pipeline for myopr-watch:
1. Fetch latest BNM data
2. Train model
3. Predict future OPR decisions
4. Save results
"""

from pathlib import Path
from datetime import datetime
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
from scripts.fetch_bnm import main as fetch_bnm_main  # 更新 opr/myor/interbank 数据
from models.train import prepare_dataset  # 训练函数
from models.predict import predict_opr, OPR_DECISIONS  # 预测函数

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

    print("[info] Predicting next OPR decision dates...")
    today = datetime.now().date()
    upcoming_dates = [d for d in OPR_DECISIONS if datetime.strptime(d, "%Y-%m-%d").date() >= today]

    predictions = []
    for fd in upcoming_dates:
        try:
            label, proba = predict_opr(fd)
            proba = {k: float(v) for k, v in proba.items()}
            predictions.append({
                "date": fd,
                "predicted_opr": label,
                **proba
            })
            print(f"Date: {fd}, Predicted OPR: {label}, Probabilities: {proba}")
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
