import os, time, joblib, httpx, difflib, shap
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import poisson
from src.feature_engineering import LiveMatchFeatureEngineer
from dotenv import load_dotenv

os.environ["GIT_PYTHON_REFRESH"] = "0"
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR, DATA_DIR = BASE_DIR / "model", BASE_DIR / "data"

app = FastAPI(title="BetCO Live Prediction Engine", version="9.0.0")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("APP_API_KEY", "fallback-dev-key"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return api_key

result_model, home_goals_model, away_goals_model, feature_pipeline, shap_explainer = None, None, None, None, None

GLOBAL_ODDS_CACHE, CACHE_TIME = [], 0

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    season: str = "2026/2027"
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None

class BatchMatchRequest(BaseModel): matches: List[MatchRequest]

@app.on_event("startup")
def load_resources():
    global result_model, home_goals_model, away_goals_model, feature_pipeline, shap_explainer
    res_path = MODEL_DIR / "prod_result_model.cbm" if (MODEL_DIR / "prod_result_model.cbm").exists() else (MODEL_DIR / "result_model.pkl" if (MODEL_DIR / "result_model.pkl").exists() else MODEL_DIR / "result.pkl")
    home_path = MODEL_DIR / "prod_home_goals_model.cbm" if (MODEL_DIR / "prod_home_goals_model.cbm").exists() else (MODEL_DIR / "Homegoal_model.pkl" if (MODEL_DIR / "Homegoal_model.pkl").exists() else MODEL_DIR / "home.pkl")
    away_path = MODEL_DIR / "prod_away_goals_model.cbm" if (MODEL_DIR / "prod_away_goals_model.cbm").exists() else (MODEL_DIR / "Awaygoal_model.pkl" if (MODEL_DIR / "Awaygoal_model.pkl").exists() else MODEL_DIR / "away.pkl")
    try:
        result_model = CatBoostClassifier().load_model(res_path) if str(res_path).endswith(".cbm") else joblib.load(res_path)
        home_goals_model = CatBoostRegressor().load_model(home_path) if str(home_path).endswith(".cbm") else joblib.load(home_path)
        away_goals_model = CatBoostRegressor().load_model(away_path) if str(away_path).endswith(".cbm") else joblib.load(away_path)
        shap_explainer = shap.TreeExplainer(result_model)
    except Exception as e: print(f"Error loading models: {e}")
    try:
        df1_path = DATA_DIR / "df1_raw.csv" if (DATA_DIR / "df1_raw.csv").exists() else BASE_DIR / "df1_raw.csv"
        team_path = DATA_DIR / "team_df_raw.csv" if (DATA_DIR / "team_df_raw.csv").exists() else BASE_DIR / "team_df_raw.csv"
        feature_pipeline = LiveMatchFeatureEngineer(pd.read_csv(df1_path), pd.read_csv(team_path))
        feature_pipeline.fit()
    except Exception as e: print(f"Error initializing pipeline: {e}")

VALID_TEAMS = ['Man City', 'West Ham', 'Middlesbrough', 'Southampton', 'Everton', 'Aston Villa', 'Bradford', 'Arsenal', 'Ipswich', 'Newcastle', 'Liverpool', 'Chelsea', 'Man United', 'Tottenham', 'Charlton', 'Sunderland', 'Derby', 'Coventry', 'Leicester', 'Leeds', 'Blackburn', 'Bolton', 'Fulham', 'West Brom', 'Birmingham', 'Wolves', 'Portsmouth', 'Crystal Palace', 'Norwich', 'Wigan', 'Watford', 'Sheffield United', 'Reading', 'Stoke', 'Hull', 'Burnley', 'Blackpool', 'Swansea', 'QPR', 'Cardiff', 'Bournemouth', 'Huddersfield', 'Brighton', 'Brentford', "Nott'm Forest", 'Luton']
TEAM_ALIASES = {"manchester united": "Man United", "man utd": "Man United", "manchester city": "Man City", "nottingham forest": "Nott'm Forest", "nottingham forest fc": "Nott'm Forest", "spurs": "Tottenham", "tottenham hotspur": "Tottenham", "wolverhampton": "Wolves", "wolverhampton wanderers": "Wolves", "newcastle united": "Newcastle", "west ham united": "West Ham", "leeds united": "Leeds", "leicester city": "Leicester", "queens park rangers": "QPR", "coventry city": "Coventry", "hull city": "Hull", "ipswich town": "Ipswich", "brighton & hove albion": "Brighton", "brighton and hove albion": "Brighton", "aston villa": "Aston Villa", "crystal palace": "Crystal Palace", "charlton athletic": "Charlton", "bolton wanderers": "Bolton", "blackburn rovers": "Blackburn", "sheffield united": "Sheffield United", "west bromwich albion": "West Brom", "west brom": "West Brom", "bristol city": "Bristol City", "luton town": "Luton", "brentford fc": "Brentford", "bournemouth": "Bournemouth", "huddersfield town": "Huddersfield"}

def standardize_team_name(input_name: str) -> str:
    if not isinstance(input_name, str): return str(input_name)
    clean = input_name.strip().lower()
    for tag in [" f.c.", " a.f.c.", " football club", " fc", " afc"]:
        if clean.endswith(tag): clean = clean[:-len(tag)].strip()
    if clean.startswith("afc "): clean = clean[4:].strip()
    if clean in TEAM_ALIASES: return TEAM_ALIASES[clean]
    for team in VALID_TEAMS:
        if clean == team.lower(): return team
    matches = difflib.get_close_matches(clean, [t.lower() for t in VALID_TEAMS], n=1, cutoff=0.6)
    if matches:
        for team in VALID_TEAMS:
            if team.lower() == matches[0]: return team
    return input_name.strip().title()

async def fetch_prematch_odds(home_team: str, away_team: str):
    global GLOBAL_ODDS_CACHE, CACHE_TIME
    key = os.getenv("ODDS_API_KEY") # Safe fallback for your convenience
    if not key: return None

    # Fetch entire EPL odds in 1 request, cache for 5 minutes
    if time.time() - CACHE_TIME > 300 or not GLOBAL_ODDS_CACHE:
        async with httpx.AsyncClient() as client:
            url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
            params = {"apiKey": key, "regions": "uk,eu", "markets": "h2h", "oddsFormat": "decimal"}
            try:
                resp = await client.get(url, params=params, timeout=10.0)
                if resp.status_code == 200:
                    GLOBAL_ODDS_CACHE = resp.json()
                    CACHE_TIME = time.time()
                else: print(f"Odds API Error: {resp.text}")
            except Exception as e: print(f"Odds API Connection Error: {e}")

    for match in GLOBAL_ODDS_CACHE:
        h_name, a_name = str(match.get("home_team", "")), str(match.get("away_team", ""))
        
        if standardize_team_name(h_name).lower() == home_team.lower() and standardize_team_name(a_name).lower() == away_team.lower():
            bookmakers = match.get("bookmakers", [])
            if not bookmakers: continue
            
            target_bookie = next((b for b in bookmakers if b.get("key") in ["pinnacle", "bet365", "betfair"]), bookmakers[0])
            for market in target_bookie.get("markets", []):
                if market.get("key") == "h2h":
                    h_odd, d_odd, a_odd = None, None, None
                    for outcome in market.get("outcomes", []):
                        name, price = outcome.get("name", ""), outcome.get("price")
                        if name == match.get("home_team"): h_odd = price
                        elif name.lower() == "draw": d_odd = price
                        elif name == match.get("away_team"): a_odd = price
                        
                    if h_odd and d_odd and a_odd:
                        return {"source": "TheOddsAPI", "bookmaker": target_bookie.get("title"), "home_odds": float(h_odd), "draw_odds": float(d_odd), "away_odds": float(a_odd), "match_id": match.get("id"), "kickoff_utc": match.get("commence_time")}
    return None

def calculate_poisson_ou(expected_goals: float, line: float):
    under_prob = poisson.cdf(int(line), expected_goals)
    return round(under_prob, 4), round(1.0 - under_prob, 4)

def determine_smart_market(home_prob, draw_prob, away_prob):
    sorted_probs = sorted({"Home Win": home_prob, "Draw": draw_prob, "Away Win": away_prob}.items(), key=lambda x: x[1], reverse=True)
    top_outcome, top_prob = sorted_probs[0]
    second_outcome, second_prob = sorted_probs[1]
    gap = top_prob - second_prob
    if gap < 0.06:
        contenders = [top_outcome, second_outcome]
        market_call = "Home or Draw (Double Chance)" if "Home Win" in contenders and "Draw" in contenders else "Away or Draw (Double Chance)" if "Away Win" in contenders and "Draw" in contenders else "Home or Away (No Draw / Close Call)"
    else: market_call = f"Outright {top_outcome}"
    return {"primary_prediction": top_outcome, "recommended_market": market_call, "confidence_gap": round(gap * 100, 1)}

def get_best_goal_tip(goal_markets):
    tips = []
    for market_type, lines in goal_markets.items():
        for line, probs in lines.items():
            if probs["Over"] >= 0.60: tips.append({"market": f"{market_type} Over {line}", "probability": round(probs["Over"], 3), "risk_level": "Low Risk 🟢" if probs["Over"] >= 0.75 else "Medium Risk 🟡" if probs["Over"] >= 0.68 else "High Risk 🔴"})
            if probs["Under"] >= 0.60: tips.append({"market": f"{market_type} Under {line}", "probability": round(probs["Under"], 3), "risk_level": "Low Risk 🟢" if probs["Under"] >= 0.75 else "Medium Risk 🟡" if probs["Under"] >= 0.68 else "High Risk 🔴"})
    tips.sort(key=lambda x: x["probability"], reverse=True)
    return tips if tips else [{"market": "Skip Goal Markets", "probability": 0.0, "risk_level": "High Risk 🔴 (No clear edge)"}]

def extract_shap_explanation(input_df, predicted_class_index):
    try:
        feature_name_map = {"HomeAttackVsAwayDefense": "Home Attacking Advantage vs Away Defense", "AwayAttackVsHomeDefense": "Away Attacking Advantage vs Home Defense", "GoalFormDiffL5": "Goal Difference Form Momentum", "FormDiff": "Overall Team Form Gap", "HomeGoalsMeanL5Venuedpd": "Home Team Scoring Form at Home", "AwayGoalsMeanL5Venuedpd": "Away Team Scoring Form Away", "HomeH2HL5": "Home Team Head-to-Head Record", "AwayH2HL5": "Away Team Head-to-Head Record", "MatchBalance": "Market Odds Parity", "era": "Historical Season Context"}
        shap_values = shap_explainer.shap_values(input_df)
        class_shap_values = shap_values[predicted_class_index] if isinstance(shap_values, list) else shap_values
        if len(class_shap_values.shape) > 1: class_shap_values = class_shap_values[0]
        feature_impacts = sorted(zip(input_df.columns, class_shap_values), key=lambda x: x[1], reverse=True)
        return [{"factor": feature_name_map.get(f, f), "impact_message": f"+{round(float(imp) * 100, 1)}% impact towards this outcome"} for f, imp in feature_impacts[:3]]
    except: return [{"factor": "Model Confidence", "impact_message": "Driven by overall team form and historical stats."}]

async def process_single_prediction(match: MatchRequest):
    home_clean, away_clean = standardize_team_name(match.home_team), standardize_team_name(match.away_team)
    if home_clean not in VALID_TEAMS or away_clean not in VALID_TEAMS: raise HTTPException(status_code=400, detail=f"Unrecognized teams: {home_clean} or {away_clean}.")
    
    odds_source, odds_bookmaker, match_id, kickoff_utc = "manual", None, None, None
    if match.home_odds is not None and match.draw_odds is not None and match.away_odds is not None:
        home_odds, draw_odds, away_odds = float(match.home_odds), float(match.draw_odds), float(match.away_odds)
    else:
        live_odds = await fetch_prematch_odds(home_clean, away_clean)
        if not live_odds: raise HTTPException(status_code=400, detail=f"Could not fetch odds for {home_clean} vs {away_clean}.")
        home_odds, draw_odds, away_odds = live_odds["home_odds"], live_odds["draw_odds"], live_odds["away_odds"]
        odds_source, odds_bookmaker, match_id, kickoff_utc = live_odds.get("source", "TheOddsAPI"), live_odds.get("bookmaker"), live_odds.get("match_id"), live_odds.get("kickoff_utc")
    
    if home_odds <= 1 or draw_odds <= 1 or away_odds <= 1: raise HTTPException(status_code=400, detail="Invalid decimal odds. Must be > 1.")
    
    input_df = feature_pipeline.transform(home_team=home_clean, away_team=away_clean, season=match.season, home_odds=home_odds, draw_odds=draw_odds, away_odds=away_odds)
    probs = result_model.predict_proba(input_df)[0]
    classes = list(result_model.classes_)
    raw_pred = result_model.predict(input_df)
    pred_val = raw_pred[0][0] if isinstance(raw_pred[0], (list, np.ndarray)) else raw_pred[0]
    predicted_class_idx = classes.index(pred_val)
    
    prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
    home_prob, draw_prob, away_prob = prob_dict.get("H", prob_dict.get("Home", 0.0)), prob_dict.get("D", prob_dict.get("Draw", 0.0)), prob_dict.get("A", prob_dict.get("Away", 0.0))
    
    exp_home_goals, exp_away_goals = max(0.01, float(home_goals_model.predict(input_df)[0])), max(0.01, float(away_goals_model.predict(input_df)[0]))
    total_exp_goals = exp_home_goals + exp_away_goals
    
    goal_markets = {
        "Total Match Goals": {"1.5": {"Under": calculate_poisson_ou(total_exp_goals, 1.5)[0], "Over": calculate_poisson_ou(total_exp_goals, 1.5)[1]}, "2.5": {"Under": calculate_poisson_ou(total_exp_goals, 2.5)[0], "Over": calculate_poisson_ou(total_exp_goals, 2.5)[1]}},
        "Home Team Goals": {"0.5": {"Under": calculate_poisson_ou(exp_home_goals, 0.5)[0], "Over": calculate_poisson_ou(exp_home_goals, 0.5)[1]}},
        "Away Team Goals": {"0.5": {"Under": calculate_poisson_ou(exp_away_goals, 0.5)[0], "Over": calculate_poisson_ou(exp_away_goals, 0.5)[1]}}
    }
    
    return {
        "match": f"{home_clean} vs {away_clean}", "match_id": match_id, "kickoff_utc": kickoff_utc,
        "odds": {"source": odds_source, "bookmaker": odds_bookmaker, "home": home_odds, "draw": draw_odds, "away": away_odds},
        "winner": str(pred_val), "market_analysis": determine_smart_market(home_prob, draw_prob, away_prob),
        "probabilities": {"HomeWin": round(home_prob, 4), "Draw": round(draw_prob, 4), "AwayWin": round(away_prob, 4)},
        "expected_goals": {"home": round(exp_home_goals, 2), "away": round(exp_away_goals, 2), "total": round(total_exp_goals, 2)},
        "smart_goal_tip": get_best_goal_tip(goal_markets), "explanation": extract_shap_explanation(input_df, predicted_class_idx)
    }

@app.get("/")
def health_check(): return {"status": "online", "message": "BetCO Engine is running.", "odds_provider": "TheOddsAPI"}

@app.post("/predict")
async def predict_match(match: MatchRequest, api_key: str = Depends(get_api_key)):
    if not result_model or not feature_pipeline: raise HTTPException(status_code=500, detail="Models not initialized.")
    return await process_single_prediction(match)

@app.post("/predict-batch")
async def predict_batch(batch: BatchMatchRequest, api_key: str = Depends(get_api_key)):
    if not result_model or not feature_pipeline: raise HTTPException(status_code=500, detail="Models not initialized.")
    results, errors = [], []
    for match in batch.matches:
        try: results.append(await process_single_prediction(match))
        except Exception as e: errors.append({"match": f"{match.home_team} vs {match.away_team}", "error": str(e)})
    return {"successful_predictions": results, "failed_predictions": errors, "summary": {"total_requested": len(batch.matches), "successful": len(results), "failed": len(errors)}}