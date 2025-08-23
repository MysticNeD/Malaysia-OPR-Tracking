# main.py
"""
Run full pipeline for myopr-watch:
1. Fetch latest BNM data
2. (Optional) retrain model
3. Predict future OPR decisions
4. Save results and optionally visualize
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import sys

# -------------- 配置路径 --------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = (ROOT / "data").resolve()
MODEL_DIR = (ROOT / "models").resolve()
OUTPUT_DIR = (ROOT / "outputs").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ 导入模块 ------------------
from scripts.fetch_bnm import main as fetch_bnm_main  # 运行 fetch_bnm.py
from models.predict import predict_opr  # 预测函数

# ------------------ 未来 OPR 决策日 ------------------
# 可以放全年 OPR 日期
OPR_DECISIONS = [
    "2025-01-22", "2025-03-06", "2025-05-08", "2025-07-09", 
    "2025-09-04", "2025-11-06"
]

# ------------------ 清理已经过去的日期 ------------------
today = datetime.now().date()
OPR_DECISIONS = [d for d in OPR_DECISIONS if datetime.strptime(d, "%Y-%m-%d").date() >= today]

# ------------------ 主流程 ------------------
def main():
    print("[info] Fetching latest BNM data...")
    try:
        fetch_bnm_main()  # 自动更新 opr/myor/interbank 数据
    except Exception as e:
        print("[error] fetch_bnm failed:", e, file=sys.stderr)

    # ------------------ 预测未来 OPR ------------------
    print("[info] Predicting for next OPR decision dates...")
    predictions = []

    for fd in OPR_DECISIONS:
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

    # ------------------ 保存 CSV ------------------
    if predictions:
        df_pred = pd.DataFrame(predictions)
        out_csv = DATA_DIR / "predictions.csv"
        df_pred.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[info] Saved predictions CSV: {out_csv} ({len(df_pred)} rows)")

    # ------------------ 保存 JSON ------------------
    out_json = DATA_DIR / "predictions.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"[info] Saved predictions JSON: {out_json}")

if __name__ == "__main__":
    main()
