# uvicorn app:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import  Request
from utils.security import verify_credentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from models.predict import predict_opr
from config import settings
import pandas as pd
import numpy as np
import os

API_KEY = os.getenv("LOAD_DATA_KEY")
app = FastAPI(debug=os.getenv("DEBUG", "false").lower() == "true")

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


# 如果会用前端（Vite 默认 5173）访问后端，请保留 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
                   "https://malaysia-opr-tracking.vercel.app",
                   "13.228.225.19",
                    "18.142.128.26",
                    "54.254.162.s138"],
    allow_credentials=True,
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

OPR_DECISIONS = [
    "2025-01-22", "2025-03-06", "2025-05-08", "2025-07-09",
    "2025-09-04", "2025-11-06"
]

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.get("/")
def read_root():
    return {"msg": "Hello World"}

@app.get("/secure-data")
def secure_data(user=Depends(verify_credentials)):
    return {"message": "This is protected"}

#@app.get("/secret-test")
#def secret_test():
#    return {"secret": settings.app_secret_key}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/predict")
def predict_next_opr(next_only: bool = True, api_key: str = Depends(verify_api_key)):
    today = datetime.now().date()
    upcoming = [d for d in OPR_DECISIONS if datetime.strptime(d, "%Y-%m-%d").date() >= today]

    if next_only and upcoming:
        upcoming = [upcoming[0]]   # ✅ 只要最早的那一个

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

# --------------------
# CSV 数据接口
# --------------------

def read_csv_as_json(path: str):
    df = pd.read_csv(path)
    df = df.replace({np.nan: 0.0})
    return df.to_dict(orient="records")

@app.get("/data/oprs")
def get_oprs(api_key: str = Depends(verify_api_key)):
    return read_csv_as_json("data/oprs.csv")

@app.get("/data/interbank_rates")
def get_interbank_rates(api_key: str = Depends(verify_api_key)):
    return read_csv_as_json("data/interbank_rates.csv")

@app.get("/data/myor")
def get_myor(api_key: str = Depends(verify_api_key)):
    return read_csv_as_json("data/myor.csv")

@app.get("/data/interbank_volumes")
def get_interbank_volumes(api_key: str = Depends(verify_api_key)):
    return read_csv_as_json("data/interbank_volumes.csv")