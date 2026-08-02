# Employee Attrition Prediction API

A FastAPI application for predicting employee attrition using a trained machine learning model. The project includes a web UI at `/`, REST endpoints, health checks, and Docker support.

## Features

- FastAPI backend with prediction endpoint
- HTML user interface served at `/`
- Health check endpoint at `/health`
- Metrics endpoint at `/metrics`
- Model info endpoint at `/model-info`
- API documentation available at `/docs`
- Dockerfile for containerized deployment

## Requirements

- Python 3.11
- `pip`
- `requirements.txt` dependencies:
  - `fastapi`
  - `pandas`
  - `joblib`
  - `uvicorn` (install separately if not included)

## Setup

1. Open a terminal in the project root.
2. (Optional but recommended) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell
source .venv/bin/activate      # macOS / Linux
```

3. Upgrade pip:

```bash
python -m pip install --upgrade pip
```

4. Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install uvicorn
```

> If `uvicorn` is missing from `requirements.txt`, install it manually with `python -m pip install uvicorn`.

## Running Locally

Start the FastAPI app with Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

If your shell cannot find `uvicorn`, use:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open:

- `http://localhost:8000/` for the UI
- `http://localhost:8000/docs` for API docs

## Docker

Build the image:

```bash
docker build -t employee-attrition-api .
```

Run the container:

```bash
docker run -p 10000:10000 employee-attrition-api
```

The Dockerfile uses the `PORT` environment variable and exposes `10000` by default.

To override the model path inside Docker:

```bash
docker run -p 10000:10000 -e MODEL_PATH=/app/model.pkl employee-attrition-api
```

## API Endpoints

- `GET /` - Serve the prediction UI
- `GET /health` - Health check and model load status
- `POST /predict` - Predict attrition from JSON payload
- `GET /metrics` - Return model metric placeholders
- `GET /model-info` - Return loaded model type and path
- `GET /docs` - OpenAPI interactive docs

### Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "DailyRate": 500,
    "DistanceFromHome": 10,
    "Education": 2,
    "EnvironmentSatisfaction": 3,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobSatisfaction": 4,
    "MaritalStatus": "Married",
    "MonthlyIncome": 5000,
    "MonthlyRate": 10000,
    "NumCompaniesWorked": 2,
    "OverTime": "No",
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3
  }'
```

## Model File

This app expects a serialized model at `model.pkl` by default. If the file is not present, the `/predict` endpoint will return a model load error.

To change the location, set the `MODEL_PATH` environment variable before running the app.

## Troubleshooting

- `bash: uvicorn: command not found`
  - Install Uvicorn with `python -m pip install uvicorn` or run `python -m uvicorn app:app ...`.

- `Model not loaded`
  - Ensure `model.pkl` exists in the project root or set `MODEL_PATH`.

- `curl http://localhost:8000/health` fails
  - Confirm the app is running and you are using the correct port.

## Notes

- The UI is rendered by FastAPI at the root route.
- The project currently uses placeholder metrics in `/metrics`; update those values after evaluating your model.
- Keep `requirements.txt` synced with any additional Python packages you install.
