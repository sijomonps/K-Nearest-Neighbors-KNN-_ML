# KNN House Price Predictor – Flask App

A web application that predicts King County house prices using a KNN regression model,
served with Flask and ready to deploy on AWS Elastic Beanstalk.

---

## Project Structure

```
KNN/
├── application.py          # Flask app (entry point for AWS EB)
├── wsgi.py                 # WSGI entry point
├── KNN.py                  # Original standalone script
├── kc_house_data.csv       # Dataset
├── requirements.txt
├── Procfile
├── .gitignore
├── .ebextensions/
│   └── python.config       # AWS Elastic Beanstalk settings
└── templates/
    ├── index.html          # Input form
    └── result.html         # Prediction result
```

> **Note:** `model.joblib` and `scaler.joblib` are auto-generated on first startup.

---

## Run Locally

```bash
# 1. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app
python application.py
# Open http://localhost:5000
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web interface (input form) |
| POST | `/predict` | HTML form submission → result page |
| POST | `/api/predict` | JSON endpoint for programmatic use |
| POST | `/retrain` | Retrain model from CSV |
| GET | `/health` | Health check |

### JSON Prediction Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3, "bathrooms": 2, "sqft_living": 1800,
    "sqft_lot": 5000, "floors": 1.5, "waterfront": 0,
    "view": 0, "condition": 3, "grade": 7,
    "sqft_above": 1500, "sqft_basement": 300,
    "yr_built": 1990, "yr_renovated": 0,
    "zipcode": 98052, "lat": 47.52, "long": -122.04,
    "sqft_living15": 1800, "sqft_lot15": 5000
  }'
```

---

## Deploy to AWS Elastic Beanstalk

### Prerequisites
- AWS CLI installed and configured (`aws configure`)
- EB CLI installed (`pip install awsebcli`)

### Steps

```bash
# 1. Initialise EB project (choose Python 3.11 platform)
eb init knn-house-predictor --platform "Python 3.11" --region us-east-1

# 2. Create the environment
eb create knn-house-env

# 3. Open the deployed app
eb open

# 4. Future deployments
eb deploy
```

### Environment Variables (optional)

Set via the EB console or CLI:

```bash
eb setenv PORT=8000
```

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | K-Nearest Neighbours Regression |
| k | 5 |
| Scaling | StandardScaler |
| Train/Test Split | 80% / 20% (random_state=42) |
| Features | 18 (after dropping `id`, `date`, `price`) |
