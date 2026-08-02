# test_raw.py
import pickle
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("TESTING MODEL WITH RAW DATA")
print("=" * 50)

# Load model
print("\nLoading model...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
print("✅ Model loaded")

# Load label encoders
print("\nLoading label encoders...")
label_encoders = joblib.load('label_encoders.pkl')
print("✅ Label encoders loaded")

# Create sample data with values that exist in training
sample = {
    "Age": 34,
    "Gender": "Male",
    "MaritalStatus": "Married",
    "DistanceFromHome": 8,
    "Education": 3,
    "EducationField": "Technical Degree",
    "JobRole": "Research Scientist",  # This exists in training
    "JobLevel": 3,
    "Department": "Research & Development",
    "BusinessTravel": "Travel_Rarely",
    "MonthlyIncome": 6500,
    "DailyRate": 500,
    "MonthlyRate": 12000,
    "TotalWorkingYears": 12,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 4,
    "NumCompaniesWorked": 2,
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "TrainingTimesLastYear": 3,
    "StockOptionLevel": 1,
    "OverTime": "No",
    "EnvironmentSatisfaction": 3,
    "JobSatisfaction": 4,
    "WorkLifeBalance": 3,
    "RelationshipSatisfaction": 3,
    "JobInvolvement": 3,
    "HourlyRate": 45
}

print("\nCreating DataFrame...")
df = pd.DataFrame([sample])
print(f"DataFrame shape: {df.shape}")

print("\nApplying label encoders...")
for col, encoder in label_encoders.items():
    if col in df.columns:
        try:
            df[col] = encoder.transform(df[col])
            print(f"  ✅ Encoded {col}")
        except Exception as e:
            print(f"  ❌ Failed to encode {col}: {e}")

print("\nConverting categorical columns...")
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.Categorical(df[col]).codes
        print(f"  ✅ Converted {col}")

print("\nConverting to numpy array...")
input_array = df.values.astype(float)
print(f"Array shape: {input_array.shape}")
print(f"Array first 5 values: {input_array[0][:5]}...")

print("\n" + "=" * 50)
print("TESTING MODEL PREDICT")
print("=" * 50)

try:
    print("Calling model.predict...")
    prediction = model.predict(input_array)
    print(f"✅ Prediction: {prediction[0]}")
    
    print("\nCalling model.predict_proba...")
    probability = model.predict_proba(input_array)
    print(f"✅ Probability: {probability[0]}")
    
    print("\n✅ SUCCESS! Model works with raw data!")
    
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()