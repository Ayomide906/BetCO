import os
os.environ["GIT_PYTHON_REFRESH"] = "0"

import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import httpx
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import poisson
import shap

# Import your feature engineer class
# [UPDATE REQUIRED]: Make sure "src.feature_engineering" matches the name of your python file in the src/ folder
from src.feature_engineering import LiveMatchFeatureEngineer

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

# Initialize FastAPI App
app = FastAPI(
    title="BetCO Live Prediction Engine",
    description="Live Premier League match outcome, Poisson goal market, and SHAP prediction engine with auto-odds fetching.",
    version="4.0.0"
)

# Global variables
result_model = None
home_goals_model = None
away_goals_model = None
feature_pipeline = None
shap_explainer = None

# --- PYDANTIC REQUEST SCHEMA ---
class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    season: str = "2025/2026"
    # Odds are optional! If frontend doesn't send them, the API will fetch them.
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None

# --- SERVER STARTUP EVENT ---
@app.on_event("startup")
def load_resources():
    """Load models, historical datasets, and initialize feature pipeline on startup."""
    global result_model, home_goals_model, away_goals_model, feature_pipeline, shap_explainer

    # 1. Path definitions (Prefers CT/CD .cbm files, falls back to Baseline .pkl files)
    res_path = MODEL_DIR / "prod_result_model.cbm" if (MODEL_DIR / "prod_result_model.cbm").exists() else (MODEL_DIR / "result_model.pkl" if (MODEL_DIR / "result_model.pkl").exists() else MODEL_DIR / "result.pkl")
    home_path = MODEL_DIR / "prod_home_goals_model.cbm" if (MODEL_DIR / "prod_home_goals_model.cbm").exists() else (MODEL_DIR / "Homegoal_model.pkl" if (MODEL_DIR / "Homegoal_model.pkl").exists() else MODEL_DIR / "home.pkl")
    away_path = MODEL_DIR / "prod_away_goals_model.cbm" if (MODEL_DIR / "prod_away_goals_model.cbm").exists() else (MODEL_DIR / "Awaygoal_model.pkl" if (MODEL_DIR / "Awaygoal_model.pkl").exists() else MODEL_DIR / "away.pkl")

    # 2. Load ML Models
    try:
        print("Loading models into API memory...")
        result_model = CatBoostClassifier().load_model(res_path) if str(res_path).endswith('.cbm') else joblib.load(res_path)
        home_goals_model = CatBoostRegressor().load_model(home_path) if str(home_path).endswith('.cbm') else joblib.load(home_path)
        away_goals_model = CatBoostRegressor().load_model(away_path) if str(away_path).endswith('.cbm') else joblib.load(away_path)

        shap_explainer = shap.TreeExplainer(result_model)
        print("✅ Models and SHAP Explainer successfully loaded.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

    # 3. Load Datasets & Fit Pipeline
    try:
        print("Pre-indexing historical datasets for feature engineering...")
        # Check data folder first, fallback to root folder if needed
        df1_path = DATA_DIR / "df1_raw.csv" if (DATA_DIR / "df1_raw.csv").exists() else BASE_DIR / "df1_raw.csv"
        team_path = DATA_DIR / "team_df_raw.csv" if (DATA_DIR / "team_df_raw.csv").exists() else BASE_DIR / "team_df_raw.csv"

        df1_raw = pd.read_csv(df1_path)
        team_df_raw = pd.read_csv(team_path)

        feature_pipeline = LiveMatchFeatureEngineer(df1_raw, team_df_raw)
        feature_pipeline.fit()
        print("✅ Feature Pipeline pre-grouped and ready for live O(1) lookups.")
    except Exception as e:
        print(f"❌ Error initializing LiveMatchFeatureEngineer: {e}")

# --- HELPER FUNCTIONS ---
async def fetch_live_bet365_odds(home_team: str, away_team: str):
    """Asynchronously fetches live odds if the user doesn't provide them."""
    
    # [UPDATE REQUIRED]: Replace this URL with the actual endpoint you used in your CI/CD pipeline!
    ODDS_API_URL = f"https://your-odds-provider.com/api/v1/bet365?home={home_team}&away={away_team}"
    
    # You can also copy/paste your actual scraping script logic right here if you aren't using an API.
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(ODDS_API_URL, timeout=5.0)
            data = response.json()
            return {
                "home_odds": data["home_odds"],
                "draw_odds": data["draw_odds"],
                "away_odds": data["away_odds"]
            }
        except Exception as e:
            print(f"Bet365 fetch failed: {e}. Using fallback odds.")
            # Fallback odds so the API doesn't crash in front of the user
            return {"home_odds": 2.50, "draw_odds": 3.20, "away_odds": 2.80}

def calculate_poisson_ou(expected_goals: float, line: float):
    k = int(line)
    under_prob = poisson.cdf(k, expected_goals)
    over_prob = 1.0 - under_prob
    return round(under_prob, 4), round(over_prob, 4)

def get_best_goal_tip(goal_markets):
    best_tip = "Skip Goal Markets (High uncertainty)"
    max_prob = 0.0

    for market_type, lines in goal_markets.items():
        for line, probs in lines.items():
            if probs["Over"] > max_prob:
                max_prob = probs["Over"]
                best_tip = f"{market_type} Over {line} (Confidence: {max_prob:.1%})"
            if probs["Under"] > max_prob:
                max_prob = probs["Under"]
                best_tip = f"{market_type} Under {line} (Confidence: {max_prob:.1%})"

    return best_tip if max_prob >= 0.70 else "Skip Goal Markets"

def extract_shap_explanation(input_df, predicted_class_index):
    # [UPDATE REQUIRED]: Feel free to add more friendly names for your 48 columns here!
    feature_name_map = {
        "HomeAttackVsAwayDefense": "Home Attacking Advantage vs Away Defense",
        "AwayAttackVsHomeDefense": "Away Attacking Advantage vs Home Defense",
        "GoalFormDiffL5": "Goal Difference Form Momentum",
        "FormDiff": "Overall Team Form Gap",
        "HomeGoalsMeanL5Venuedpd": "Home Team Scoring Form at Home",
        "AwayGoalsMeanL5Venuedpd": "Away Team Scoring Form Away",
        "HomeH2HL5": "Home Team Head-to-Head Record",
        "AwayH2HL5": "Away Team Head-to-Head Record",
        "MatchBalance": "Market Odds Parity",
        "era": "Historical Season Context"
    }

    shap_values = shap_explainer.shap_values(input_df)
    class_shap_values = shap_values[predicted_class_index][0]

    feature_impacts = sorted(list(zip(input_df.columns, class_shap_values)), key=lambda x: x[1], reverse=True)

    top_reasons = []
    for feature, impact in feature_impacts[:3]:
        friendly_name = feature_name_map.get(feature, feature)
        confidence_boost = round(float(impact) * 100, 1)
        top_reasons.append({
            "factor": friendly_name,
            "impact_message": f"+{confidence_boost}% impact towards this outcome"
        })

    return top_reasons

# --- ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "online", "message": "BetCO Live Prediction Engine Operational"}

@app.post("/predict")
async def predict_match(match: MatchRequest):
    if not result_model or not feature_pipeline:
        raise HTTPException(status_code=500, detail="Models or feature pipeline not initialized.")

    # 1. Fetch Odds if Missing (Non-blocking I/O)
    if match.home_odds is None or match.draw_odds is None or match.away_odds is None:
        print(f"Fetching live Bet365 odds for {match.home_team} vs {match.away_team}...")
        live_odds = await fetch_live_bet365_odds(match.home_team, match.away_team)
        home_odds = live_odds["home_odds"]
        draw_odds = live_odds["draw_odds"]
        away_odds = live_odds["away_odds"]
    else:
        home_odds = match.home_odds
        draw_odds = match.draw_odds
        away_odds = match.away_odds

    try:
        # 2. Generate 48 Features On The Fly (CPU-bound Math)
        input_df = feature_pipeline.transform(
            home_team=match.home_team,
            away_team=match.away_team,
            season=match.season,
            home_odds=home_odds,
            draw_odds=draw_odds,
            away_odds=away_odds
        )

        # 3. Predict Match Outcome
        probs = result_model.predict_proba(input_df)[0]
        classes = list(result_model.classes_)
        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}

        home_prob = prob_dict.get('H', prob_dict.get('Home', 0.0))
        draw_prob = prob_dict.get('D', prob_dict.get('Draw', 0.0))
        away_prob = prob_dict.get('A', prob_dict.get('Away', 0.0))

        predicted_class_str = str(result_model.predict(input_df)[0])
        predicted_class_idx = classes.index(predicted_class_str)

        # 4. Predict Expected Goals
        exp_home_goals = max(0.01, float(home_goals_model.predict(input_df)[0]))
        exp_away_goals = max(0.01, float(away_goals_model.predict(input_df)[0]))
        total_exp_goals = exp_home_goals + exp_away_goals

        # 5. Compute Poisson Goal Markets
        goal_markets = {
            "Total Match Goals": {
                "1.5": {"Under": calculate_poisson_ou(total_exp_goals, 1.5)[0], "Over": calculate_poisson_ou(total_exp_goals, 1.5)[1]},
                "2.5": {"Under": calculate_poisson_ou(total_exp_goals, 2.5)[0], "Over": calculate_poisson_ou(total_exp_goals, 2.5)[1]}
            },
            "Home Team Goals": {
                "0.5": {"Under": calculate_poisson_ou(exp_home_goals, 0.5)[0], "Over": calculate_poisson_ou(exp_home_goals, 0.5)[1]},
                "1.5": {"Under": calculate_poisson_ou(exp_home_goals, 1.5)[0], "Over": calculate_poisson_ou(exp_home_goals, 1.5)[1]}
            },
            "Away Team Goals": {
                "0.5": {"Under": calculate_poisson_ou(exp_away_goals, 0.5)[0], "Over": calculate_poisson_ou(exp_away_goals, 0.5)[1]},
                "1.5": {"Under": calculate_poisson_ou(exp_away_goals, 1.5)[0], "Over": calculate_poisson_ou(exp_away_goals, 1.5)[1]}
            }
        }

        # 6. Extract SHAP Explanation
        explanation = extract_shap_explanation(input_df, predicted_class_idx)

        # 7. Generate Smart Tip
        best_goal_tip = get_best_goal_tip(goal_markets)

        return {
            "match_info": {
                "home_team": match.home_team,
                "away_team": match.away_team,
                "market_odds_used": {
                    "home_win": home_odds,
                    "draw": draw_odds,
                    "away_win": away_odds
                }
            },
            "prediction": {
                "winner": predicted_class_str,
                "probabilities": {
                    "HomeWin": round(home_prob, 4),
                    "Draw": round(draw_prob, 4),
                    "AwayWin": round(away_prob, 4)
                }
            },
            "expected_goals": {
                "home": round(exp_home_goals, 2),
                "away": round(exp_away_goals, 2),
                "total": round(total_exp_goals, 2)
            },
            "goal_markets": goal_markets,
            "smart_goal_tip": best_goal_tip,
            "model_explanation": {
                "message": f"Primary drivers for predicting {predicted_class_str}:",
                "top_driving_factors": explanation
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction pipeline failed: {str(e)}")