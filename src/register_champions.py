import joblib
import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier, CatBoostRegressor
from pathlib import Path
import os

# 1. Silence GitPython warnings before importing MLflow
os.environ["GIT_PYTHON_REFRESH"] = "0"

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

# SQLite DB Path for MLflow tracking
MLFLOW_DB_PATH = DATA_DIR / "mlflow.db"

def load_model_file(path_options):
    """Helper to flexibly load models whether saved as .pkl via joblib or .cbm via CatBoost."""
    for path in path_options:
        if path.exists():
            if path.suffix == '.pkl':
                return joblib.load(path)
            elif path.suffix == '.cbm':
                # Native Catboost loading
                if "result" in path.name.lower():
                    return CatBoostClassifier().load_model(path)
                else:
                    return CatBoostRegressor().load_model(path)
                    
    raise FileNotFoundError(f"Could not find any of these models in {MODEL_DIR}: {[p.name for p in path_options]}")

def register_league_models(league_name, experiment_name, paths, metrics):
    print(f"\n--- Registering {league_name} Champions ---")
    
    clf_model = load_model_file(paths['result'])
    hg_model = load_model_file(paths['home'])
    ag_model = load_model_file(paths['away'])

    # Setup MLflow with SQLite backend
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"Baseline_{league_name}_Champions") as run:
        print(f"Logging {league_name} models to MLflow registry...")
        
        # Log models into MLflow registry
        mlflow.catboost.log_model(clf_model, name=f"{league_name.lower()}_result_model")
        mlflow.catboost.log_model(hg_model, name=f"{league_name.lower()}_home_goals_model")
        mlflow.catboost.log_model(ag_model, name=f"{league_name.lower()}_away_goals_model")
        
        # Log baseline metrics
        mlflow.log_metrics(metrics)
        
        mlflow.set_tag("source", "notebook_baseline")
        mlflow.set_tag("status", "production")
        mlflow.set_tag("league", league_name)

        print(f"✅ Success! {league_name} Champions logged to MLflow.")
        print(f"Run ID: {run.info.run_id}")


def register_existing_champions():
    # ==========================================
    # 1. EPL CONFIGURATION
    # ==========================================
    epl_paths = {
        'result': [MODEL_DIR / "result_model.pkl", MODEL_DIR / "result.pkl"],
        'home': [MODEL_DIR / "Homegoal_model.pkl", MODEL_DIR / "home.pkl"],
        'away': [MODEL_DIR / "Awaygoal_model.pkl", MODEL_DIR / "away.pkl"]
    }
    epl_metrics = {
        "result_accuracy": 0.517,
        "home_goals_rmse": 1.236,
        "away_goals_rmse": 1.143,
        "away_goals_accuracy": 0.362,
        "home_goals_accuracy": 0.311
    }
    
    # ==========================================
    # 2. LA LIGA CONFIGURATION
    # ==========================================
    # Checking for .cbm and .pkl depending on how you saved them in the notebook
    laliga_paths = {
        'result': [MODEL_DIR / "laliga_result_model.pkl", MODEL_DIR / "laliga_result_model.cbm", MODEL_DIR / "laligaresult_model.pkl"],
        'home': [MODEL_DIR / "laliga_home_goals_model.pkl", MODEL_DIR / "laliga_home_goals_model.cbm", MODEL_DIR / "laligaHomegoal_model.pkl"],
        'away': [MODEL_DIR / "laliga_away_goals_model.pkl", MODEL_DIR / "laliga_away_goals_model.cbm", MODEL_DIR / "laligaAwaygoal_model.pkl"]
    }
    laliga_metrics = {
        "result_accuracy": 0.494,
        "home_goals_rmse": 1.18,  
        "away_goals_rmse": 1.03   
    }

    # Register both leagues!
    register_league_models("EPL", "Premier_League_Predictor", epl_paths, epl_metrics)
    register_league_models("LaLiga", "LaLiga_Predictor", laliga_paths, laliga_metrics)

if __name__ == "__main__":
    register_existing_champions()