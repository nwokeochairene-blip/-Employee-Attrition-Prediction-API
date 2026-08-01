from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

# Load your model
try:
    model = joblib.load('model.pkl')  # or whatever your model file is named
except:
    # Handle case where model doesn't exist
    model = None

class EmployeeData(BaseModel):
    # Define your input features here
    # Example:
    Age: int
    DailyRate: int
    # ... add all your features

@app.get("/")
async def root():
    return {"message": "Employee Attrition Prediction API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(data: EmployeeData):
    if model is None:
        return {"error": "Model not loaded"}
    # Convert to dataframe and predict
    # input_data = pd.DataFrame([data.dict()])
    # prediction = model.predict(input_data)
    # return {"prediction": prediction.tolist()}
    return {"message": "Prediction endpoint"}