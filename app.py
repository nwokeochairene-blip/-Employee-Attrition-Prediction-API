# app.py - FINAL WORKING VERSION (No Scaler)
import pickle
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmployeeData(BaseModel):
    Age: int
    Gender: str
    MaritalStatus: str
    DistanceFromHome: float
    Education: int
    EducationField: str
    JobRole: str
    JobLevel: int
    Department: str
    BusinessTravel: str
    MonthlyIncome: float
    DailyRate: float
    MonthlyRate: float
    TotalWorkingYears: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int
    NumCompaniesWorked: int
    PercentSalaryHike: float
    PerformanceRating: int
    TrainingTimesLastYear: int
    StockOptionLevel: int
    OverTime: str
    EnvironmentSatisfaction: int
    JobSatisfaction: int
    WorkLifeBalance: int
    RelationshipSatisfaction: int
    JobInvolvement: int
    HourlyRate: float

model = None
label_encoders = None

def load_model_files():
    global model, label_encoders
    
    try:
        # Load model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
        
        # Load label encoders
        try:
            label_encoders = joblib.load('label_encoders.pkl')
            print("✅ Label encoders loaded!")
            print(f"Encoded columns: {list(label_encoders.keys())}")
        except Exception as e:
            label_encoders = None
            print(f"⚠️ Label encoders not loaded: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    success = load_model_files()
    if success:
        print("✅ API is ready!")
    else:
        print("⚠️ API started but model not loaded")

@app.get("/")
async def root():
    return {
        "status": "ready" if model is not None else "model_not_loaded",
        "message": "Employee Attrition Prediction API",
        "model_loaded": model is not None,
        "encoders_loaded": label_encoders is not None
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(data: EmployeeData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        
        print(f"Input data: {input_dict}")
        
        # Apply label encoding if available
        if label_encoders:
            for col, encoder in label_encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = encoder.transform(input_df[col])
                        print(f"✅ Encoded {col}")
                    except Exception as e:
                        print(f"⚠️ Could not encode {col}: {e}")
                        # Handle unseen labels
                        if col == 'JobRole':
                            # Map common job roles to existing ones
                            job_role_mapping = {
                                'Data Scientist': 'Research Scientist',
                                'Data Analyst': 'Research Scientist', 
                                'Software Engineer': 'Research Scientist',
                                'ML Engineer': 'Research Scientist',
                                'AI Engineer': 'Research Scientist',
                                'Business Analyst': 'Sales Executive',
                                'Product Manager': 'Manager',
                                'Project Manager': 'Manager',
                                'Team Lead': 'Manager',
                                'Developer': 'Research Scientist'
                            }
                            value = input_df[col].iloc[0]
                            if value in job_role_mapping:
                                mapped_value = job_role_mapping[value]
                                input_df[col] = encoder.transform([mapped_value])
                                print(f"✅ Mapped {value} to {mapped_value}")
                            else:
                                input_df[col] = 0
                                print(f"⚠️ Using 0 for unknown job role: {value}")
                        else:
                            input_df[col] = 0
        
        # Convert all categorical to numeric
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                input_df[col] = pd.Categorical(input_df[col]).codes
                print(f"✅ Converted {col} to numeric")
        
        # Ensure all columns are numeric
        input_df = input_df.astype(float)
        
        # Convert to numpy array (NO SCALING - LightGBM doesn't need it)
        input_array = input_df.values
        
        print(f"Input array shape: {input_array.shape}")
        print(f"Input array: {input_array}")
        
        # Make prediction
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)
        
        print(f"Prediction: {prediction[0]}")
        print(f"Probability: {probability[0]}")
        
        return {
            "prediction": "Yes" if prediction[0] == 1 else "No",
            "probability_attrition": float(probability[0][1]),
            "prediction_code": int(prediction[0]),
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)