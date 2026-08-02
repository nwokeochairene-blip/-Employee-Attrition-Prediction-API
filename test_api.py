# test_api.py
import requests
import json

url = "http://localhost:8000/predict"

data = {
    "Age": 34,
    "Gender": "Male",
    "MaritalStatus": "Married",
    "DistanceFromHome": 8,
    "Education": 3,
    "EducationField": "Technical Degree",
    "JobRole": "Data Scientist",
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

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")