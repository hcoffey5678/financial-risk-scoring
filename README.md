# Financial Risk Scoring API

A financial risk scoring machine learning project built with FastAPI, XGBoost, TensorFlow, Scikit-Learn, MLflow, and Docker-ready design. It predicts loan default risk and supports:

- Machine learning model training & comparison
- Explainability with SHAP
- Model serving with FastAPI
- MLflow experiment tracking
- Health check endpoints

## Features

- XGBoost and TensorFlow models for credit risk scoring
- Random Forest benchmark model
- Model versioning with MLflow
- API serving with FastAPI (XGBoost and TensorFlow endpoints)
- Health check API
- Full dataset EDA (Exploratory Data Analysis)
- ROC Curve comparison
- CORS enabled for frontend use

## Tech Stack

| Component | Tech Used |
|:--|:--|
| API Framework | FastAPI |
| Language | Python 3.10 |
| Machine Learning | Scikit-Learn, XGBoost, TensorFlow |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Container | Docker (optional setup) |
| Testing | Jupyter Notebooks, FastAPI Swagger |

# Project Structure

```
risk_scoring_project/
├── app/
│   └── fastapi_app.py
├── data/
│   ├── lending_club_data.csv
│   └── processed/
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
├── models/
│   ├── xgboost_model_best.pkl
│   ├── tensorflow_model.h5
│   └── random_forest_pipeline.pkl
├── notebooks/
│   ├── 01_basic_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training_rf.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_mlflow_tracking.ipynb
│   ├── 06_tensorflow_model_training.ipynb
│   ├── 07_compare_models_mlflow.ipynb
│   └── 08_api_prediction_demo.ipynb
├── requirements.txt
├── Dockerfile (optional)
└── README.md

```

## Getting Started

### Prerequisites
- Python 3.10+
- pip
- (Optional) Docker

### Local Dev Setup
```bash
git clone https://github.com/hcoffey5678/financial-risk-scoring.git
cd financial-risk-scoring
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.fastapi_app:app --reload --port 8000
```

### Run MLflow UI (Optional)
```bash
mlflow ui --port 5001
```
Access MLflow Tracking UI at:
http://localhost:5001

## Run Tests (Notebook Based)
- Open Jupyter
- Run notebooks in `/notebooks`
- Alternatively test live predictions with `/predict/sample`

## API Endpoints (Docs)
Once running, access interactive Swagger UI at:

http://localhost:8000/docs

### Authentication
Not required for now (public endpoints). Future versions may add JWT auth.

## Environment Variables
Key | Description
--- | ---
MLFLOW_TRACKING_URI | Path for MLflow experiments (defaults to ./mlruns)

## License
MIT

## Contributing
PRs are welcome! Please open an issue first to discuss changes or improvements.

## Acknowledgements
Inspired by real-world fintech applications for regulation-compliant AI modeling.
