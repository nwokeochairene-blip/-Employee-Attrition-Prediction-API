from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Optional
import uvicorn
import os  # <-- ADD THIS IMPORT

FEATURE_COLUMNS = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

CATEGORICAL_COLUMNS = [
    'BusinessTravel', 'Department', 'EducationField', 'Gender',
    'JobRole', 'MaritalStatus', 'OverTime'
]

NUMERICAL_COLUMNS = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

DEFAULT_VALUES = {
    'JobSatisfaction': 3,
    'MaritalStatus': 'Married',
    'WorkLifeBalance': 3,
    'DailyRate': 500,
    'DistanceFromHome': 10,
    'EducationField': 'Life Sciences',
    'EnvironmentSatisfaction': 3,
    'HourlyRate': 50,
    'JobInvolvement': 3,
    'JobLevel': 1,
    'MonthlyRate': 5000,
    'NumCompaniesWorked': 2,
    'PercentSalaryHike': 15,
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 3,
    'StockOptionLevel': 1,
    'TotalWorkingYears': 10,
    'TrainingTimesLastYear': 2,
    'YearsInCurrentRole': 4,
    'YearsSinceLastPromotion': 2,
    'YearsWithCurrManager': 4
}

try:
    model = joblib.load('attrition_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
except Exception as exc:
    raise RuntimeError(f'Failed to load model artifacts: {exc}') from exc

app = FastAPI(
    title='Employee Attrition Prediction API',
    description='API for predicting employee attrition risk',
    version='1.0'
)

# Define input data model
class EmployeeData(BaseModel):
    Age: int
    BusinessTravel: str
    Department: str
    Education: int
    Gender: str
    JobRole: str
    MonthlyIncome: float
    OverTime: str
    YearsAtCompany: int
    
    # Optional additional features (will use defaults if not provided)
    JobSatisfaction: Optional[int] = 3
    MaritalStatus: Optional[str] = "Married"
    WorkLifeBalance: Optional[int] = 3
    DailyRate: Optional[float] = 500
    DistanceFromHome: Optional[int] = 10
    EducationField: Optional[str] = "Life Sciences"
    EnvironmentSatisfaction: Optional[int] = 3
    HourlyRate: Optional[float] = 50
    JobInvolvement: Optional[int] = 3
    JobLevel: Optional[int] = 1
    MonthlyRate: Optional[float] = 5000
    NumCompaniesWorked: Optional[int] = 2
    PercentSalaryHike: Optional[float] = 15
    PerformanceRating: Optional[int] = 3
    RelationshipSatisfaction: Optional[int] = 3
    StockOptionLevel: Optional[int] = 1
    TotalWorkingYears: Optional[int] = 10
    TrainingTimesLastYear: Optional[int] = 2
    YearsInCurrentRole: Optional[int] = 4
    YearsSinceLastPromotion: Optional[int] = 2
    YearsWithCurrManager: Optional[int] = 4

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Employee Attrition Prediction API",
        "endpoints": {
            "/": "This information",
            "/health": "Health check",
            "/predict": "POST endpoint for attrition prediction"
        },
        "sample_request": {
            "Age": 35,
            "BusinessTravel": "Travel_Rarely",
            "Department": "Research & Development",
            "Education": 3,
            "Gender": "Male",
            "JobRole": "Research Scientist",
            "MonthlyIncome": 6000,
            "OverTime": "No",
            "YearsAtCompany": 5
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict")
async def predict_attrition(employee: EmployeeData):
    """
    Predict employee attrition probability
    
    Returns:
    - prediction: "Yes" or "No"
    - probability: Confidence score (0-1)
    """
    try:
        # Convert input to dictionary and fill missing values
        input_dict = employee.dict()
        for field, default_value in DEFAULT_VALUES.items():
            if input_dict.get(field) is None:
                input_dict[field] = default_value
        
        # Create DataFrame
        input_df = pd.DataFrame([employee.model_dump()])
        input_df = input_df.reindex(columns=FEATURE_COLUMNS)

        for col in FEATURE_COLUMNS:
            if col in DEFAULT_VALUES:
                input_df[col] = input_df[col].fillna(DEFAULT_VALUES[col])
            elif col in NUMERICAL_COLUMNS:
                input_df[col] = input_df[col].fillna(0)
            else:
                input_df[col] = input_df[col].fillna("")

        for col in CATEGORICAL_COLUMNS:
            if col not in label_encoders:
                raise HTTPException(status_code=500, detail=f'Missing label encoder for column: {col}')

            encoder = label_encoders[col]
            values = input_df[col].astype(str)
            unknown_mask = ~values.isin(encoder.classes_)
            if unknown_mask.any():
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown categories for '{col}': {values[unknown_mask].tolist()}"
                )
            input_df[col] = encoder.transform(values)

        input_df[NUMERICAL_COLUMNS] = scaler.transform(input_df[NUMERICAL_COLUMNS].astype(float))

        prediction = model.predict(input_df)
        probability = 1.0
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)
            classes = list(model.classes_)
            predicted_class = prediction[0]
            if predicted_class in classes:
                probability = float(proba[0][classes.index(predicted_class)])
            elif 1 in classes:
                probability = float(proba[0][classes.index(1)])
            elif 'Yes' in classes:
                probability = float(proba[0][classes.index('Yes')])
            else:
                probability = float(proba[0].max())

        result = 'Yes' if str(prediction[0]) in {'1', 'Yes', 'yes', 'Y', 'y'} else 'No'

        return {
            'prediction': result,
            'probability': round(probability, 4)
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Error making prediction: {exc}')

if __name__ == "__main__":
    # CHANGE THIS LINE - use Render's PORT environment variable
    port = int(os.environ.get("PORT", 8000))  # Defaults to 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)