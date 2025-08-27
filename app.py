# uvicorn app:app --host 0.0.0.0 --port 8000
# Updated app.py file
# This file combines all API endpoints into a single FastAPI application
# to simplify deployment and avoid Vercel-specific runtime configuration issues.

from fastapi import FastAPI, Depends, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from config import settings
import pandas as pd
import numpy as np
import os

# Define the API key for authentication
API_KEY = os.getenv("LOAD_DATA_KEY")

# Create a FastAPI app instance
app = FastAPI(debug=os.getenv("DEBUG", "false").lower() == "true")

# Print the API key to the console for debugging purposes
print("LOAD_DATA_KEY:", API_KEY)

# Define a function to verify the API key from the request header
def verify_api_key(x_api_key: str = Header(None), request: Request = None):
    if request.method == "OPTIONS":
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

# Configure CORS (Cross-Origin Resource Sharing) middleware
# This is crucial for allowing the frontend (on a different domain) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
                   "https://malaysia-opr-tracking.vercel.app",
                   "https://malaysia-opr-tracking.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Combined API Endpoints
# All data and prediction endpoints are now in one file
# --------------------

# Health check endpoint
@app.get("/health")
@app.head("/health")
def health():
    return Response(content='{"ok": true}', media_type="application/json")



# Utility function to read CSV data into a JSON-like format
def read_csv_as_json(path: str):
    """
    Reads a CSV file from a given path and converts it to a list of dictionaries (JSON format).
    It also handles missing values (NaN) by replacing them with 0.0.
    """
    try:
        df = pd.read_csv(path)
        df = df.replace({np.nan: 0.0})
        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

# Endpoint to get OPR (Overnight Policy Rate) data from oprs.csv
@app.get("/data/oprs")
def get_oprs(api_key: str = Depends(verify_api_key)):
    return read_csv_as_json("data/oprs.csv")

# Endpoint to get MYOR (Malaysia Overnight Rate) data from myor.csv
@app.get("/data/myor")
def get_myor(api_key: str = Depends(verify_api_key)):
    return read_csv_as_json("data/myor.csv")

# Endpoint to get interbank rates data from interbank_rates.csv
@app.get("/data/interbank_rates")
def get_interbank_rates(api_key: str = Depends(verify_api_key)):
    return read_csv_as_json("data/interbank_rates.csv")

# Endpoint to get interbank volumes data from interbank_volumes.csv
@app.get("/data/interbank_volumes")
def get_interbank_volumes(api_key: str = Depends(verify_api_key)):
    return read_csv_as_json("data/interbank_volumes.csv")

# Define OPR decision dates and a dummy prediction function
# This part of the code needs to be adapted from your original project's models/predict.py
# For now, it's a simple placeholder
OPR_DECISIONS = [
    "2025-01-01",
    "2025-03-05",
    "2025-05-07",
    "2025-07-09",
    "2025-09-11",
    "2025-11-06",
]

def predict_opr(date_str):
    """
    A placeholder function to simulate OPR prediction.
    In your original code, this would be a more complex model.
    """
    # Dummy logic to simulate a prediction
    from random import uniform
    increase_prob = uniform(0, 0.3)
    decrease_prob = uniform(0, 0.3)
    hold_prob = 1.0 - increase_prob - decrease_prob
    
    probabilities = {
        "increase": increase_prob,
        "decrease": decrease_prob,
        "hold": hold_prob,
    }
    
    predicted_label = max(probabilities, key=probabilities.get)
    return predicted_label, probabilities

# Endpoint to get the next OPR prediction
@app.get("/predict")
def predict_next_opr(request: Request, next_only: bool = True, api_key: str = Depends(verify_api_key)):
    print("REQUEST HEADERS:", request.headers)
    today = datetime.now().date()
    upcoming = [d for d in OPR_DECISIONS if datetime.strptime(d, "%Y-%m-%d").date() >= today]

    if next_only and upcoming:
        upcoming = [upcoming[0]]

    results = []
    for fd in upcoming:
        label, proba = predict_opr(fd)
        proba = {k: float(v) for k, v in proba.items()}
        results.append({
            "date": fd,
            "predicted_opr": label,
            "probabilities": proba
        })
    print(results)
    return results

# Serve static files for the frontend, assuming a "dist" directory
# This is for hosting the entire app on a single server like Render or on Vercel
# with a build script that creates a "dist" folder.
# app.mount("/", StaticFiles(directory="dist", html=True), name="static")

