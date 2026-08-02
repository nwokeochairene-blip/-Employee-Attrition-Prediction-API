from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Employee Attrition Prediction API")

# Define the EmployeeData model with ALL 30 features
class EmployeeData(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

# Load model and encoders
model = None
label_encoders = None
scaler = None
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")

# Load label encoders if they exist
try:
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    logger.info("Label encoders loaded successfully")
except Exception as e:
    logger.warning(f"Label encoders not loaded: {e}")

# Load scaler if it exists
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully")
except Exception as e:
    logger.warning(f"Scaler not loaded: {e}")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the HTML interface"""
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the root directory.</p>")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.post("/predict")
async def predict(data: EmployeeData):
    """Predict employee attrition based on input features"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # If you have label encoders, apply them
        if label_encoders:
            for col in label_encoders:
                if col in input_df.columns:
                    input_df[col] = label_encoders[col].transform(input_df[col])
        
        # If you have a scaler, apply it
        if scaler:
            # Get numeric columns (exclude categorical ones)
            numeric_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[0]
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        return {
            "prediction": int(prediction),
            "probability": probability,
            "confidence": probability if probability else None
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get model metrics (placeholder)"""
    return {
        "accuracy": 0.85,
        "precision": 0.83,
        "recall": 0.81,
        "f1_score": 0.82
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_type": str(type(model).__name__) if model else None,
        "features_expected": 30  # Updated to 30
    }