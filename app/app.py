import os
os.environ["GIT_PYTHON_REFRESH"] = "0"

from dotenv import load_dotenv
load_dotenv()

import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import httpx
import difflib
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import poisson
import shap

from src.feature_engineering import LiveMatchFeatureEngineer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

app = FastAPI(
    title="BetCO Live Prediction Engine",
    description="Live match outcome, Poisson goal market, market gap analysis, and batch prediction engine with fuzzy string matching.",
    version="6.4.0"
)

result_model = None
home_goals_model = None
away_goals_model = None
feature_pipeline = None
shap_explainer = None

# Securely load API key from environment variables
STATS_API_KEY = os.getenv("STATS_API_KEY", "")
STATS_API_BASE = "https://api.thestatsapi.com/api/football"

# --- PYDANTIC SCHEMAS ---
class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    season: str = "2025/2026"
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None

class GoalTip(BaseModel):
    market: str
    probability: float
    risk_level: str

class BatchMatchRequest(BaseModel):
    matches: List[MatchRequest]

# --- STARTUP EVENT ---
@app.on_event("startup")
def load_resources():
    global result_model, home_goals_model, away_goals_model, feature_pipeline, shap_explainer

    res_path = MODEL_DIR / "prod_result_model.cbm" if (MODEL_DIR / "prod_result_model.cbm").exists() else (MODEL_DIR / "result_model.pkl" if (MODEL_DIR / "result_model.pkl").exists() else MODEL_DIR / "result.pkl")
    home_path = MODEL_DIR / "prod_home_goals_model.cbm" if (MODEL_DIR / "prod_home_goals_model.cbm").exists() else (MODEL_DIR / "Homegoal_model.pkl" if (MODEL_DIR / "Homegoal_model.pkl").exists() else MODEL_DIR / "home.pkl")
    away_path = MODEL_DIR / "prod_away_goals_model.cbm" if (MODEL_DIR / "prod_away_goals_model.cbm").exists() else (MODEL_DIR / "Awaygoal_model.pkl" if (MODEL_DIR / "Awaygoal_model.pkl").exists() else MODEL_DIR / "away.pkl")

    try:
        print("Loading models into API memory...")
        result_model = CatBoostClassifier().load_model(res_path) if str(res_path).endswith('.cbm') else joblib.load(res_path)
        home_goals_model = CatBoostRegressor().load_model(home_path) if str(home_path).endswith('.cbm') else joblib.load(home_path)
        away_goals_model = CatBoostRegressor().load_model(away_path) if str(away_path).endswith('.cbm') else joblib.load(away_path)
        shap_explainer = shap.TreeExplainer(result_model)
        print("✅ Models and SHAP Explainer successfully loaded.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

    try:
        print("Pre-indexing historical datasets...")
        df1_path = DATA_DIR / "df1_raw.csv" if (DATA_DIR / "df1_raw.csv").exists() else BASE_DIR / "df1_raw.csv"
        team_path = DATA_DIR / "team_df_raw.csv" if (DATA_DIR / "team_df_raw.csv").exists() else BASE_DIR / "team_df_raw.csv"

        df1_raw = pd.read_csv(df1_path)
        team_df_raw = pd.read_csv(team_path)

        feature_pipeline = LiveMatchFeatureEngineer(df1_raw, team_df_raw)
        feature_pipeline.fit()
        print("✅ Feature Pipeline ready.")
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")

VALID_TEAMS = [
    'Man City', 'West Ham', 'Middlesbrough', 'Southampton', 'Everton',
    'Aston Villa', 'Bradford', 'Arsenal', 'Ipswich', 'Newcastle',
    'Liverpool', 'Chelsea', 'Man United', 'Tottenham', 'Charlton',
    'Sunderland', 'Derby', 'Coventry', 'Leicester', 'Leeds',
    'Blackburn', 'Bolton', 'Fulham', 'West Brom', 'Birmingham',
    'Wolves', 'Portsmouth', 'Crystal Palace', 'Norwich', 'Wigan',
    'Watford', 'Sheffield United', 'Reading', 'Stoke', 'Hull',
    'Burnley', 'Blackpool', 'Swansea', 'QPR', 'Cardiff', 'Bournemouth',
    'Huddersfield', 'Brighton', 'Brentford', "Nott'm Forest", 'Luton'
]

TEAM_ALIASES = {
    "manchester united": "Man United",
    "man utd": "Man United",
    "manchester city": "Man City",
    "nottingham forest": "Nott'm Forest",
    "spurs": "Tottenham",
    "wolverhampton": "Wolves",
    "wolverhampton wanderers": "Wolves",
    "newcastle united": "Newcastle",
    "west ham united": "West Ham",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "queens park rangers": "QPR"
}

def standardize_team_name(input_name: str) -> str:
    clean_input = input_name.strip().lower()
    if clean_input in TEAM_ALIASES:
        return TEAM_ALIASES[clean_input]
    for valid_team in VALID_TEAMS:
        if clean_input == valid_team.lower():
            return valid_team
    matches = difflib.get_close_matches(clean_input, [t.lower() for t in VALID_TEAMS], n=1, cutoff=0.6)
    if matches:
        for valid_team in VALID_TEAMS:
            if valid_team.lower() == matches[0]:
                return valid_team
    return input_name.strip().title()

# --- LIVE STATS API ODDS FETCHING WITH PARSING & FALLBACK ---
async def fetch_live_bet365_odds(home_team: str, away_team: str):
    headers = {"Authorization": f"Bearer {STATS_API_KEY}"}
    url = f"{STATS_API_BASE}/matches?status=scheduled&per_page=50"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                matches_list = data.get("data", data.get("matches", []))
                
                for match in matches_list:
                    h = standardize_team_name(match.get("homeTeam", match.get("home_team", "")))
                    a = standardize_team_name(match.get("awayTeam", match.get("away_team", "")))
                    
                    if h.lower() == home_team.lower() and a.lower() == away_team.lower():
                        odds = match.get("odds", {})
                        return {
                            "home_odds": float(odds.get("home", 2.0)),
                            "draw_odds": float(odds.get("draw", 3.2)),
                            "away_odds": float(odds.get("away", 3.5))
                        }
            return None
        except Exception as e:
            print(f"TheStatsAPI connection warning: {e}")
            return None

def calculate_poisson_ou(expected_goals: float, line: float):
    under_prob = poisson.cdf(int(line), expected_goals)
    return round(under_prob, 4), round(1.0 - under_prob, 4)

def determine_smart_market(home_prob, draw_prob, away_prob):
    probs = {"Home Win": home_prob, "Draw": draw_prob, "Away Win": away_prob}
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    top_outcome, top_prob = sorted_probs[0]
    second_outcome, second_prob = sorted_probs[1]
    
    gap = top_prob - second_prob
    
    if gap < 0.06:
        contenders = sorted([top_outcome, second_outcome])
        if "Home Win" in contenders and "Draw" in contenders:
            market_call = "Home or Draw (Double Chance)"
        elif "Away Win" in contenders and "Draw" in contenders:
            market_call = "Away or Draw (Double Chance)"
        else:
            market_call = "Home or Away (No Draw / Close Call)"
    else:
        market_call = f"Outright {top_outcome}"
        
    return {
        "primary_prediction": top_outcome,
        "recommended_market": market_call,
        "confidence_gap": round(gap * 100, 1)
    }

def get_best_goal_tip(goal_markets):
    tips = []
    for market_type, lines in goal_markets.items():
        for line, probs in lines.items():
            over_prob = probs["Over"]
            if over_prob >= 0.60:
                risk_tag = "Low Risk 🟢" if over_prob >= 0.75 else "Medium Risk 🟡" if over_prob >= 0.68 else "High Risk 🔴"
                tips.append({"market": f"{market_type} Over {line}", "probability": round(over_prob, 3), "risk_level": risk_tag})
            under_prob = probs["Under"]
            if under_prob >= 0.60:
                risk_tag = "Low Risk 🟢" if over_prob >= 0.75 else "Medium Risk 🟡" if over_prob >= 0.68 else "High Risk 🔴"
                tips.append({"market": f"{market_type} Under {line}", "probability": round(under_prob, 3), "risk_level": risk_tag})
                
    tips = sorted(tips, key=lambda x: x["probability"], reverse=True)
    if not tips:
        return [{"market": "Skip Goal Markets", "probability": 0.0, "risk_level": "High Risk 🔴 (No clear edge)"}]
    return tips

def extract_shap_explanation(input_df, predicted_class_index):
    try:
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
        class_shap_values = shap_values[predicted_class_index] if isinstance(shap_values, list) else shap_values
        if len(class_shap_values.shape) > 1:
            class_shap_values = class_shap_values[0]

        feature_impacts = sorted(list(zip(input_df.columns, class_shap_values)), key=lambda x: x[1], reverse=True)
        top_reasons = []
        for feature, impact in feature_impacts[:3]:
            friendly_name = feature_name_map.get(feature, feature)
            confidence_boost = round(float(impact) * 100, 1)
            top_reasons.append({"factor": friendly_name, "impact_message": f"+{confidence_boost}% impact towards this outcome"})
        return top_reasons
    except Exception as e:
        print(f"SHAP warning: {e}")
        return [{"factor": "Model Confidence", "impact_message": "Driven by overall team form and historical stats."}]

async def process_single_prediction(match: MatchRequest):
    home_clean = standardize_team_name(match.home_team)
    away_clean = standardize_team_name(match.away_team)

    if home_clean not in VALID_TEAMS or away_clean not in VALID_TEAMS:
        raise HTTPException(status_code=400, detail=f"Unrecognized teams: {home_clean} or {away_clean}. Allowed: {', '.join(VALID_TEAMS)}")

    if match.home_odds is None or match.draw_odds is None or match.away_odds is None:
        live_odds = await fetch_live_bet365_odds(home_clean, away_clean)
        if not live_odds:
            raise HTTPException(status_code=400, detail=f"Could not automatically fetch live odds for {home_clean} vs {away_clean}. Please provide home_odds, draw_odds, and away_odds manually in your request.")
        home_odds, draw_odds, away_odds = live_odds["home_odds"], live_odds["draw_odds"], live_odds["away_odds"]
    else:
        home_odds, draw_odds, away_odds = match.home_odds, match.draw_odds, match.away_odds

    input_df = feature_pipeline.transform(
        home_team=home_clean, away_team=away_clean, season=match.season,
        home_odds=home_odds, draw_odds=draw_odds, away_odds=away_odds
    )

    probs = result_model.predict_proba(input_df)[0]
    classes = list(result_model.classes_)
    
    raw_pred = result_model.predict(input_df)
    predicted_class_str = str(raw_pred[0][0]) if isinstance(raw_pred[0], (list, np.ndarray)) else str(raw_pred[0])
    predicted_class_idx = classes.index(predicted_class_str)

    prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
    home_prob = prob_dict.get('H', prob_dict.get('Home', 0.0))
    draw_prob = prob_dict.get('D', prob_dict.get('Draw', 0.0))
    away_prob = prob_dict.get('A', prob_dict.get('Away', 0.0))

    market_analysis = determine_smart_market(home_prob, draw_prob, away_prob)

    exp_home_goals = max(0.01, float(home_goals_model.predict(input_df)[0]))
    exp_away_goals = max(0.01, float(away_goals_model.predict(input_df)[0]))
    total_exp_goals = exp_home_goals + exp_away_goals

    goal_markets = {
        "Total Match Goals": {
            "1.5": {"Under": calculate_poisson_ou(total_exp_goals, 1.5)[0], "Over": calculate_poisson_ou(total_exp_goals, 1.5)[1]},
            "2.5": {"Under": calculate_poisson_ou(total_exp_goals, 2.5)[0], "Over": calculate_poisson_ou(total_exp_goals, 2.5)[1]}
        },
        "Home Team Goals": {
            "0.5": {"Under": calculate_poisson_ou(exp_home_goals, 0.5)[0], "Over": calculate_poisson_ou(exp_home_goals, 0.5)[1]}
        },
        "Away Team Goals": {
            "0.5": {"Under": calculate_poisson_ou(exp_away_goals, 0.5)[0], "Over": calculate_poisson_ou(exp_away_goals, 0.5)[1]}
        }
    }

    return {
        "match": f"{home_clean} vs {away_clean}",
        "winner": predicted_class_str,
        "market_analysis": market_analysis,
        "probabilities": {"HomeWin": round(home_prob, 4), "Draw": round(draw_prob, 4), "AwayWin": round(away_prob, 4)},
        "expected_goals": {"home": round(exp_home_goals, 2), "away": round(exp_away_goals, 2), "total": round(total_exp_goals, 2)},
        "smart_goal_tip": get_best_goal_tip(goal_markets),
        "explanation": extract_shap_explanation(input_df, predicted_class_idx)
    }

@app.get("/")
def health_check():
    return {"status": "online", "message": "BetCO Engine is running."}

@app.post("/predict")
async def predict_match(match: MatchRequest):
    if not result_model or not feature_pipeline:
        raise HTTPException(status_code=500, detail="Models or pipeline not initialized.")
    try:
        return await process_single_prediction(match)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(batch: BatchMatchRequest):
    if not result_model or not feature_pipeline:
        raise HTTPException(status_code=500, detail="Models or pipeline not initialized.")
    
    results = []
    errors = []
    for match in batch.matches:
        try:
            result = await process_single_prediction(match)
            results.append(result)
        except Exception as e:
            errors.append({"match": f"{match.home_team} vs {match.away_team}", "error": str(e)})
            
    return {"successful_predictions": results, "failed_predictions": errors}