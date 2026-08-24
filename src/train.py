import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.catboost
import os
import argparse

# 1. Silence GitPython warnings before importing MLflow
os.environ["GIT_PYTHON_REFRESH"] = "0"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model" 

# ==========================================
# MLOps SETTINGS
# ==========================================
# The new model must be at least 1% better to replace the current production model.
MIN_IMPROVEMENT_THRESHOLD = 0.01 

def train_and_evaluate(league_name="EPL"):
    print(f"\n🚀 Starting Training Pipeline for {league_name}...")
    
    # 1. Load engineered dataset specific to the league
    # (Assuming your data prep script saves them as X_train_epl.csv, X_train_laliga.csv, etc.)
    # If not, you can adjust these filenames!
    x_path = DATA_DIR / f"X_train_{league_name.lower()}.csv"
    y_path = DATA_DIR / f"y_train_{league_name.lower()}.csv"
    
    if not x_path.exists() or not y_path.exists():
        print(f"❌ Could not find data for {league_name} at {x_path}. Run feature engineering first!")
        return
        
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)

    # ==========================================
    # 2. TIME-BASED 3-WAY SPLIT (Train/Val/Test)
    # ==========================================
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train_all, y_val_all, y_test_all = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
    
    # Separate targets
    y_train_result, y_val_result, y_test_result = y_train_all['FullTimeResult'], y_val_all['FullTimeResult'], y_test_all['FullTimeResult']
    y_train_h_goals, y_val_h_goals, y_test_h_goals = y_train_all['FullTimeHomeGoals'], y_val_all['FullTimeHomeGoals'], y_test_all['FullTimeHomeGoals']
    y_train_a_goals, y_val_a_goals, y_test_a_goals = y_train_all['FullTimeAwayGoals'], y_val_all['FullTimeAwayGoals'], y_test_all['FullTimeAwayGoals']

    # 3. Setup MLflow for the specific league
    mlflow.set_tracking_uri(f"sqlite:///{DATA_DIR}/mlflow.db") # Using the SQLite DB we set up earlier!
    
    # Dynamic experiment name (e.g., "Premier_League_Predictor" or "LaLiga_Predictor")
    experiment_name = "Premier_League_Predictor" if league_name.upper() == "EPL" else f"{league_name}_Predictor"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"Automated_Training_{league_name}") as run:
        cat_features = ['HomeTeam', 'AwayTeam']
        
        # ==========================================
        # MODEL 1: MATCH RESULT (CLASSIFIER)
        # ==========================================
        print(f"[{league_name}] Training Result Classifier...")
        clf_model = CatBoostClassifier(
            iterations=2000, depth=4, learning_rate=0.03, l2_leaf_reg=5,
            random_strength=2, auto_class_weights='Balanced', bagging_temperature=5,
            loss_function="MultiClass", eval_metric="MultiClass", early_stopping_rounds=50,
            random_seed=42, verbose=0
        )
        clf_model.fit(X_train, y_train_result, eval_set=(X_val, y_val_result), cat_features=cat_features)
        
        clf_preds = clf_model.predict(X_test)
        clf_probs = clf_model.predict_proba(X_test)
        acc = accuracy_score(y_test_result, clf_preds)
        loss = log_loss(y_test_result, clf_probs)
        
        # ==========================================
        # MODEL 2: HOME GOALS (REGRESSOR)
        # ==========================================
        print(f"[{league_name}] Training Home Goals Regressor...")
        hg_model = CatBoostRegressor(
            iterations=3000, learning_rate=0.01, depth=5, loss_function="Poisson",
            eval_metric="Poisson", random_strength=5, bagging_temperature=3,
            random_seed=42, early_stopping_rounds=100, verbose=0
        )
        hg_model.fit(X_train, y_train_h_goals, eval_set=(X_val, y_val_h_goals), cat_features=cat_features)
        
        hg_preds = hg_model.predict(X_test)
        hg_rmse = np.sqrt(mean_squared_error(y_test_h_goals, hg_preds))
        hg_mae = mean_absolute_error(y_test_h_goals, hg_preds)
        
        # ==========================================
        # MODEL 3: AWAY GOALS (REGRESSOR)
        # ==========================================
        print(f"[{league_name}] Training Away Goals Regressor...")
        ag_model = CatBoostRegressor(
            iterations=3000, learning_rate=0.01, depth=5, loss_function="Poisson",
            eval_metric="Poisson", random_strength=5, bagging_temperature=3,
            random_seed=42, early_stopping_rounds=100, verbose=0
        )
        ag_model.fit(X_train, y_train_a_goals, eval_set=(X_val, y_val_a_goals), cat_features=cat_features)
        
        ag_preds = ag_model.predict(X_test)
        ag_rmse = np.sqrt(mean_squared_error(y_test_a_goals, ag_preds))
        ag_mae = mean_absolute_error(y_test_a_goals, ag_preds)

        # ==========================================
        # LOGGING
        # ==========================================
        print(f"\n--- {league_name} NEW MODEL RESULTS (TEST DATA) ---")
        print(f"Result Accuracy: {acc:.4f} | Log Loss: {loss:.4f}")
        print(f"Home Goals RMSE: {hg_rmse:.4f} | MAE: {hg_mae:.4f}")
        print(f"Away Goals RMSE: {ag_rmse:.4f} | MAE: {ag_mae:.4f}")
        
        mlflow.log_metrics({
            "result_accuracy": acc, "result_log_loss": loss,
            "home_goals_rmse": hg_rmse, "home_goals_mae": hg_mae,
            "away_goals_rmse": ag_rmse, "away_goals_mae": ag_mae
        })
        
        mlflow.catboost.log_model(clf_model, f"{league_name.lower()}_result_model")
        mlflow.catboost.log_model(hg_model, f"{league_name.lower()}_home_goals_model")
        mlflow.catboost.log_model(ag_model, f"{league_name.lower()}_away_goals_model")
        
        # ==========================================
        # CHALLENGER VS CHAMPION COMPARISON
        # ==========================================
        MODEL_DIR.mkdir(exist_ok=True)
        print("\n--- EVALUATING AGAINST CURRENT PRODUCTION MODELS ---")

        # Dynamic paths for production models based on league
        prefix = "prod" if league_name.upper() == "EPL" else f"prod_{league_name.lower()}"
        prod_clf_path = MODEL_DIR / f"{prefix}_result_model.cbm"
        prod_hg_path = MODEL_DIR / f"{prefix}_home_goals_model.cbm"
        prod_ag_path = MODEL_DIR / f"{prefix}_away_goals_model.cbm"

        # 1. Result Model (Log Loss)
        update_clf = True
        if prod_clf_path.exists():
            try:
                old_clf = CatBoostClassifier().load_model(prod_clf_path)
                old_loss = log_loss(y_test_result, old_clf.predict_proba(X_test))
                improvement = (old_loss - loss) / old_loss
                print(f"Result -> Old Loss: {old_loss:.4f} | New Loss: {loss:.4f}")
                if improvement < MIN_IMPROVEMENT_THRESHOLD:
                    print(f"❌ Result model improvement ({improvement:.2%}) below threshold. Keeping Champion.")
                    update_clf = False
                else:
                    print(f"✅ Result model improved by {improvement:.2%}! Promoting to Champion.")
            except Exception as e:
                print(f"⚠️ Could not load old Result model: {e}. Overwriting.")

        if update_clf:
            clf_model.save_model(prod_clf_path)

        # 2. Home Goals (RMSE)
        update_hg = True
        if prod_hg_path.exists():
            try:
                old_hg = CatBoostRegressor().load_model(prod_hg_path)
                old_hg_rmse = np.sqrt(mean_squared_error(y_test_h_goals, old_hg.predict(X_test)))
                improvement = (old_hg_rmse - hg_rmse) / old_hg_rmse
                print(f"Home Goals -> Old RMSE: {old_hg_rmse:.4f} | New RMSE: {hg_rmse:.4f}")
                if improvement < MIN_IMPROVEMENT_THRESHOLD:
                    print(f"❌ Home model improvement ({improvement:.2%}) below threshold. Keeping Champion.")
                    update_hg = False
                else:
                    print(f"✅ Home model improved by {improvement:.2%}! Promoting to Champion.")
            except Exception: pass

        if update_hg: hg_model.save_model(prod_hg_path)

        # 3. Away Goals (RMSE)
        update_ag = True
        if prod_ag_path.exists():
            try:
                old_ag = CatBoostRegressor().load_model(prod_ag_path)
                old_ag_rmse = np.sqrt(mean_squared_error(y_test_a_goals, old_ag.predict(X_test)))
                improvement = (old_ag_rmse - ag_rmse) / old_ag_rmse
                print(f"Away Goals -> Old RMSE: {old_ag_rmse:.4f} | New RMSE: {ag_rmse:.4f}")
                if improvement < MIN_IMPROVEMENT_THRESHOLD:
                    print(f"❌ Away model improvement ({improvement:.2%}) below threshold. Keeping Champion.")
                    update_ag = False
                else:
                    print(f"✅ Away model improved by {improvement:.2%}! Promoting to Champion.")
            except Exception: pass

        if update_ag: ag_model.save_model(prod_ag_path)
            
        print(f"\n🎉 {league_name} pipeline complete!")

if __name__ == "__main__":
    # Setup argparse so you can run: python src/train.py --league LaLiga
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=str, default="EPL", help="League to train (EPL, LaLiga, etc.)")
    args = parser.parse_args()
    
    train_and_evaluate(args.league)