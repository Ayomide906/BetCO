import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.catboost
import os

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

def train_and_evaluate():
    print("Loading engineered dataset...")
    X = pd.read_csv(DATA_DIR / "X_train.csv")
    y = pd.read_csv(DATA_DIR / "y_train.csv")

    # ==========================================
    # 1. TIME-BASED 3-WAY SPLIT (Train/Val/Test)
    # ==========================================
    # Train (70%) - Used to fit the model
    # Validation (15%) - Used for Early Stopping (CatBoost eval_set)
    # Test (15%) - Strictly unseen data for Champion vs Challenger
    
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    # Feature splits
    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    
    # Target splits (All)
    y_train_all = y.iloc[:train_end]
    y_val_all = y.iloc[train_end:val_end]
    y_test_all = y.iloc[val_end:]
    
    # Separate the specific targets
    y_train_result = y_train_all['FullTimeResult']
    y_val_result = y_val_all['FullTimeResult']
    y_test_result = y_test_all['FullTimeResult']
    
    y_train_h_goals = y_train_all['FullTimeHomeGoals']
    y_val_h_goals = y_val_all['FullTimeHomeGoals']
    y_test_h_goals = y_test_all['FullTimeHomeGoals']
    
    y_train_a_goals = y_train_all['FullTimeAwayGoals']
    y_val_a_goals = y_val_all['FullTimeAwayGoals']
    y_test_a_goals = y_test_all['FullTimeAwayGoals']

    # 2. Setup MLflow
    mlflow.set_tracking_uri(f"file://{BASE_DIR}/mlruns")
    mlflow.set_experiment("Premier_League_Predictor")
    
    with mlflow.start_run() as run:
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
            eval_metric="MultiClass",
            early_stopping_rounds=50,
            random_seed=42,
            verbose=0
        )
        # Fit using Val set for early stopping
        clf_model.fit(X_train, y_train_result, eval_set=(X_val, y_val_result), cat_features=cat_features)
        
        # Evaluate exclusively on Test set
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
        hg_model.fit(X_train, y_train_h_goals, eval_set=(X_val, y_val_h_goals), cat_features=cat_features)
        
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
        ag_model.fit(X_train, y_train_a_goals, eval_set=(X_val, y_val_a_goals), cat_features=cat_features)
        
        ag_preds = ag_model.predict(X_test)
        ag_rmse = np.sqrt(mean_squared_error(y_test_a_goals, ag_preds))
        ag_mae = mean_absolute_error(y_test_a_goals, ag_preds)
        ag_r2 = r2_score(y_test_a_goals, ag_preds)

        # ==========================================
        # LOGGING
        # ==========================================
        print(f"\n--- NEW MODEL RESULTS (ON UNSEEN TEST DATA) ---")
        print(f"Result Accuracy: {acc:.4f} | Log Loss: {loss:.4f}")
        print(f"Home Goals RMSE: {hg_rmse:.4f} | MAE: {hg_mae:.4f}")
        print(f"Away Goals RMSE: {ag_rmse:.4f} | MAE: {ag_mae:.4f}")
        
        mlflow.log_metrics({
            "result_accuracy": acc,
            "result_log_loss": loss,
            "home_goals_rmse": hg_rmse,
            "home_goals_mae": hg_mae,
            "home_goals_r2": hg_r2,
            "away_goals_rmse": ag_rmse,
            "away_goals_mae": ag_mae,
            "away_goals_r2": ag_r2
        })
        
        mlflow.catboost.log_model(clf_model, "result_model")
        mlflow.catboost.log_model(hg_model, "home_goals_model")
        mlflow.catboost.log_model(ag_model, "away_goals_model")
        
        # ==========================================
        # CHALLENGER VS CHAMPION COMPARISON
        # ==========================================
        MODEL_DIR.mkdir(exist_ok=True)
        print("\n--- EVALUATING AGAINST CURRENT PRODUCTION MODELS ---")

        # 1. Result Model (Log Loss Threshold - Lower is better)
        prod_clf_path = MODEL_DIR / "prod_result_model.cbm"
        update_clf = True
        if prod_clf_path.exists():
            try:
                old_clf = CatBoostClassifier().load_model(prod_clf_path)
                old_probs = old_clf.predict_proba(X_test)
                old_loss = log_loss(y_test_result, old_probs)
                
                print(f"Result Model -> Old Log Loss: {old_loss:.4f} | New Log Loss: {loss:.4f}")
                
                improvement = (old_loss - loss) / old_loss
                if improvement < MIN_IMPROVEMENT_THRESHOLD:
                    print(f"❌ Improvement ({improvement:.2%}) is below {MIN_IMPROVEMENT_THRESHOLD:.2%} threshold. Keeping Champion.")
                    update_clf = False
                else:
                    print(f"✅ New Result model achieved meaningful improvement ({improvement:.2%})! Promoting to Champion.")
            except Exception as e:
                print(f"⚠️ Could not load old Result model: {e}. Overwriting with new model.")

        if update_clf:
            clf_model.save_model(prod_clf_path)
            print("💾 Production Result model updated.")

        # 2. Home Goals (RMSE Threshold - Lower is better)
        prod_hg_path = MODEL_DIR / "prod_home_goals_model.cbm"
        update_hg = True
        if prod_hg_path.exists():
            try:
                old_hg = CatBoostRegressor().load_model(prod_hg_path)
                old_hg_preds = old_hg.predict(X_test)
                old_hg_rmse = np.sqrt(mean_squared_error(y_test_h_goals, old_hg_preds))
                
                print(f"Home Goals -> Old RMSE: {old_hg_rmse:.4f} | New RMSE: {hg_rmse:.4f}")
                
                improvement = (old_hg_rmse - hg_rmse) / old_hg_rmse
                if improvement < MIN_IMPROVEMENT_THRESHOLD:
                    print(f"❌ Improvement ({improvement:.2%}) is below {MIN_IMPROVEMENT_THRESHOLD:.2%} threshold. Keeping Champion.")
                    update_hg = False
                else:
                    print(f"✅ New Home Goals model achieved meaningful improvement ({improvement:.2%})! Promoting to Champion.")
            except Exception as e:
                print(f"⚠️ Could not load old Home Goals model: {e}. Overwriting with new model.")

        if update_hg:
            hg_model.save_model(prod_hg_path)
            print("💾 Production Home Goals model updated.")

        # 3. Away Goals (RMSE Threshold - Lower is better)
        prod_ag_path = MODEL_DIR / "prod_away_goals_model.cbm"
        update_ag = True
        if prod_ag_path.exists():
            try:
                old_ag = CatBoostRegressor().load_model(prod_ag_path)
                old_ag_preds = old_ag.predict(X_test)
                old_ag_rmse = np.sqrt(mean_squared_error(y_test_a_goals, old_ag_preds))
                
                print(f"Away Goals -> Old RMSE: {old_ag_rmse:.4f} | New RMSE: {ag_rmse:.4f}")
                
                improvement = (old_ag_rmse - ag_rmse) / old_ag_rmse
                if improvement < MIN_IMPROVEMENT_THRESHOLD:
                    print(f"❌ Improvement ({improvement:.2%}) is below {MIN_IMPROVEMENT_THRESHOLD:.2%} threshold. Keeping Champion.")
                    update_ag = False
                else:
                    print(f"✅ New Away Goals model achieved meaningful improvement ({improvement:.2%})! Promoting to Champion.")
            except Exception as e:
                print(f"⚠️ Could not load old Away Goals model: {e}. Overwriting with new model.")

        if update_ag:
            ag_model.save_model(prod_ag_path)
            print("💾 Production Away Goals model updated.")
            
        print("\n🎉 Training pipeline complete!")

if __name__ == "__main__":
    train_and_evaluate()