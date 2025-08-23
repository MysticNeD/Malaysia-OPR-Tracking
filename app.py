from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from models.predict import predict_opr

app = FastAPI(title="MY OPR Watch API")

# 如果会用前端（Vite 默认 5173）访问后端，请保留 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPR_DECISIONS = [
    "2025-01-22", "2025-03-06", "2025-05-08", "2025-07-09",
    "2025-09-04", "2025-11-06"
]

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/predict")
def predict_next_opr(next_only: bool = False):
    today = datetime.now().date()
    upcoming = [d for d in OPR_DECISIONS if datetime.strptime(d, "%Y-%m-%d").date() >= today]
    if next_only and upcoming:
        upcoming = [upcoming[0]]  # 只预测下一次

    results = []
    for fd in upcoming:
        label, proba = predict_opr(fd)
        proba = {k: float(v) for k, v in proba.items()}
        results.append({
            "date": fd,
            "predicted_opr": label,
            "probabilities": proba
        })
    return results
