import joblib
import mlflow
import mlflow.catboost
from pathlib import Path
import os
# 1. Silence GitPython warnings before importing MLflow
os.environ["GIT_PYTHON_REFRESH"] = "0"
# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

# SQLite DB Path for MLflow tracking (Fixes Windows path issues)
MLFLOW_DB_PATH = DATA_DIR / "mlflow.db"

def register_existing_champions():
    print("Loading existing champion models...")
    
    # 1. Load models (using joblib as saved in your notebook)
    # Check for both naming conventions
    result_path = MODEL_DIR / "result_model.pkl" if (MODEL_DIR / "result_model.pkl").exists() else MODEL_DIR / "result.pkl"
    home_path = MODEL_DIR / "Homegoal_model.pkl" if (MODEL_DIR / "Homegoal_model.pkl").exists() else MODEL_DIR / "home.pkl"
    away_path = MODEL_DIR / "Awaygoal_model.pkl" if (MODEL_DIR / "Awaygoal_model.pkl").exists() else MODEL_DIR / "away.pkl"

    clf_model = joblib.load(result_path)
    hg_model = joblib.load(home_path)
    ag_model = joblib.load(away_path)

    # 2. Setup MLflow with SQLite backend
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("Premier_League_Predictor")
    
    # 3. Log baseline run
    with mlflow.start_run(run_name="Baseline_Notebook_Champions") as run:
        print("Logging models to MLflow...")
        
        # Log models into MLflow registry
        mlflow.catboost.log_model(clf_model, name="result_model")
        mlflow.catboost.log_model(hg_model, name="home_goals_model")
        mlflow.catboost.log_model(ag_model, name="away_goals_model")
        
        # Log baseline metrics from your notebook
        mlflow.log_metrics({
            "result_accuracy": 0.517,
            "home_goals_rmse": 1.236,
            "away_goals_rmse": 1.143,
            "away_goals_accuracy": 0.362,
            "home_goals_accuracy": 0.311
        })
        
        mlflow.set_tag("source", "notebook_baseline")
        mlflow.set_tag("status", "production")

        print(f"\nSuccess! Baseline Champions logged to MLflow.")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    register_existing_champions()