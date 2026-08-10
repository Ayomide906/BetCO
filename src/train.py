import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, mean_absolute_error,r2_score
import mlflow
import mlflow.catboost
import os
# 1. Silence GitPython warnings before importing MLflow
os.environ["GIT_PYTHON_REFRESH"] = "0"

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model" 

def train_and_evaluate():
    print("Loading engineered dataset...")
    X = pd.read_csv(DATA_DIR / "X_train.csv")
    y = pd.read_csv(DATA_DIR / "y_train.csv")

    # 1. Time-Based Split (85% Train, 15% Test)
    split_index = int(len(X) * 0.85)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train_all, y_test_all = y.iloc[:split_index], y.iloc[split_index:]
    
    # Separate the targets
    y_train_result = y_train_all['FullTimeResult']
    y_test_result = y_test_all['FullTimeResult']
    
    y_train_h_goals = y_train_all['FullTimeHomeGoals']
    y_test_h_goals = y_test_all['FullTimeHomeGoals']
    
    y_train_a_goals = y_train_all['FullTimeAwayGoals']
    y_test_a_goals = y_test_all['FullTimeAwayGoals']

    # 2. Setup MLflow
    mlflow.set_tracking_uri(f"file://{BASE_DIR}/mlruns")
    mlflow.set_experiment("Premier_League_Predictor")
    
    with mlflow.start_run() as run:
        # Define categorical features
        cat_features = ['HomeTeam', 'AwayTeam']
        
        # ==========================================
        # MODEL 1: MATCH RESULT (CLASSIFIER)
        # ==========================================
        print("Training Result Classifier...")
        clf_model = CatBoostClassifier(
            iterations=2000,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=5,
            random_strength=2,
            auto_class_weights='Balanced',
            bagging_temperature=5,
            loss_function="MultiClass",
            #class_weights=[1.25, 1.5, 1],
            eval_metric="MultiClass",
            early_stopping_rounds=50,
            random_seed=42,
            verbose=0
            )
        clf_model.fit(X_train, y_train_result, eval_set=(X_test, y_test_result))
        
        clf_preds = clf_model.predict(X_test)
        clf_probs = clf_model.predict_proba(X_test)
        
        acc = accuracy_score(y_test_result, clf_preds)
        loss = log_loss(y_test_result, clf_probs)
        
        # ==========================================
        # MODEL 2: HOME GOALS (REGRESSOR)
        # ==========================================
        print("Training Home Goals Regressor...")
        hg_model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.01,
            depth=5,
            loss_function="Poisson",
            eval_metric="Poisson",
            random_strength=5,
            bagging_temperature=3,
            random_seed=42,
            early_stopping_rounds=100,
            verbose=0
            )
        hg_model.fit(X_train, y_train_h_goals, eval_set=(X_test, y_test_h_goals))
        
        hg_preds = hg_model.predict(X_test)
        hg_rmse = np.sqrt(mean_squared_error(y_test_h_goals, hg_preds))
        hg_mae = mean_absolute_error(y_test_h_goals, hg_preds)
        hg_r2 = r2_score(y_test_h_goals, hg_preds)
        
        # ==========================================
        # MODEL 3: AWAY GOALS (REGRESSOR)
        # ==========================================
        print("Training Away Goals Regressor...")
        ag_model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.01,
            depth=5,
            loss_function="Poisson",
            eval_metric="Poisson",
            random_strength=5,
            bagging_temperature=3,
            random_seed=42,
            early_stopping_rounds=100,
            verbose=0
            )
        ag_model.fit(X_train, y_train_a_goals, eval_set=(X_test, y_test_a_goals))
        
        ag_preds = ag_model.predict(X_test)
        ag_rmse = np.sqrt(mean_squared_error(y_test_a_goals, ag_preds))
        ag_mae = mean_absolute_error(y_test_a_goals, ag_preds)
        ag_r2 = r2_score(y_test_a_goals, ag_preds)

        # ==========================================
        # LOGGING & SAVING
        # ==========================================
        print(f"\n--- RESULTS ---")
        print(f"Result Accuracy: {acc:.4f} | Log Loss: {loss:.4f}")
        print(f"Home Goals RMSE: {hg_rmse:.4f} | MAE: {hg_mae:.4f}")
        print(f"Away Goals RMSE: {ag_rmse:.4f} | MAE: {ag_mae:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "result_accuracy": acc,
            "result_log_loss": loss,
            "home_goals_rmse": hg_rmse,
            "home_goals_mae": hg_mae,
            "home_goals_r2":hg_r2,
            "away_goals_rmse": ag_rmse,
            "away_goals_mae": ag_mae,
            "away_goals_r2":ag_r2
            })
        
        # Save models to MLflow (for version history)
        mlflow.catboost.log_model(clf_model, "result_model")
        mlflow.catboost.log_model(hg_model, "home_goals_model")
        mlflow.catboost.log_model(ag_model, "away_goals_model")
        
        # Save hard copies for your API
        MODEL_DIR.mkdir(exist_ok=True)
        clf_model.save_model(MODEL_DIR / "prod_result_model.cbm")
        hg_model.save_model(MODEL_DIR / "prod_home_goals_model.cbm")
        ag_model.save_model(MODEL_DIR / "prod_away_goals_model.cbm")
        
        print(f"\nProduction models updated in: {MODEL_DIR}")

if __name__ == "__main__":
    train_and_evaluate()