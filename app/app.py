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

app = FastAPI(title="BetCO Live Prediction Engine", version="10.0.0")
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

async def fetch_prematch_odds(home_team: str, away_team: str):
    global GLOBAL_ODDS_CACHE, CACHE_TIME
    key = os.getenv("ODDS_API_KEY")
    if not key: return None
    if time.time() - CACHE_TIME > 300 or not GLOBAL_ODDS_CACHE:
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get("https://api.the-odds-api.com/v4/sports/soccer_epl/odds", params={"apiKey": key, "regions": "uk,eu", "markets": "h2h", "oddsFormat": "decimal"}, timeout=10.0)
                if resp.status_code == 200: GLOBAL_ODDS_CACHE, CACHE_TIME = resp.json(), time.time()
            except: pass
    for match in GLOBAL_ODDS_CACHE:
        if standardize_team_name(match.get("home_team", "")).lower() == home_team.lower() and standardize_team_name(match.get("away_team", "")).lower() == away_team.lower():
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

def poisson_over_probability(expected_goals: float, line: float) -> float: return round(float(poisson.sf(int(line), expected_goals)), 4)
def poisson_under_probability(expected_goals: float, line: float) -> float: return round(float(poisson.cdf(int(line), expected_goals)), 4)
def build_ou_markets(expected_goals: float, lines=(0.5, 1.5, 2.5, 3.5, 4.5, 5.5)): return {str(line): {"Under": poisson_under_probability(expected_goals, line), "Over": poisson_over_probability(expected_goals, line)} for line in lines}

def build_score_matrix(home_xg: float, away_xg: float, max_goals: int = 6):
    return [{"home_goals": h, "away_goals": a, "probability": float(poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg))} for h in range(max_goals + 1) for a in range(max_goals + 1)]

def calculate_btts(home_xg: float, away_xg: float):
    btts_yes = (1 - poisson.pmf(0, home_xg)) * (1 - poisson.pmf(0, away_xg))
    return {"Yes": round(float(btts_yes), 4), "No": round(float(1 - btts_yes), 4)}

def calculate_clean_sheets(home_xg: float, away_xg: float): return {"Home Clean Sheet": round(float(poisson.pmf(0, away_xg)), 4), "Away Clean Sheet": round(float(poisson.pmf(0, home_xg)), 4)}
def get_correct_scores(score_matrix, top_n=5): return [{"score": f"{x['home_goals']}-{x['away_goals']}", "probability": round(x["probability"], 4)} for x in sorted(score_matrix, key=lambda x: x["probability"], reverse=True)[:top_n]]
def calculate_double_chance(home_prob, draw_prob, away_prob): return {"1X": round(home_prob + draw_prob, 4), "X2": round(draw_prob + away_prob, 4), "12": round(home_prob + away_prob, 4)}

def calculate_draw_no_bet(home_prob, draw_prob, away_prob):
    non_draw = home_prob + away_prob
    return {"Home DNB": round(home_prob / non_draw, 4), "Away DNB": round(away_prob / non_draw, 4)} if non_draw > 0 else {"Home DNB": 0.0, "Away DNB": 0.0}

def calculate_score_markets(score_matrix, home_team, away_team):
    m = {k: 0.0 for k in ["h_o15", "h_o25", "a_o15", "a_o25", "h_btts", "a_btts", "h_nil", "a_nil", "h_2", "a_2"]}
    for item in score_matrix:
        h, a, p, total = item["home_goals"], item["away_goals"], item["probability"], item["home_goals"] + item["away_goals"]
        if h > a:
            if total >= 2: m["h_o15"] += p
            if total >= 3: m["h_o25"] += p
            if a >= 1: m["h_btts"] += p
            if a == 0: m["h_nil"] += p
            if h - a >= 2: m["h_2"] += p
        elif a > h:
            if total >= 2: m["a_o15"] += p
            if total >= 3: m["a_o25"] += p
            if h >= 1: m["a_btts"] += p
            if h == 0: m["a_nil"] += p
            if a - h >= 2: m["a_2"] += p
    return {"Home Win + Over 1.5": round(m["h_o15"], 4), "Home Win + Over 2.5": round(m["h_o25"], 4), "Away Win + Over 1.5": round(m["a_o15"], 4), "Away Win + Over 2.5": round(m["a_o25"], 4), "Home Win + BTTS": round(m["h_btts"], 4), "Away Win + BTTS": round(m["a_btts"], 4), "Home Win To Nil": round(m["h_nil"], 4), "Away Win To Nil": round(m["a_nil"], 4), "Home Win By 2+": round(m["h_2"], 4), "Away Win By 2+": round(m["a_2"], 4)}

def get_risk_level(probability): return "Low Risk 🟢" if probability >= 0.80 else "Medium-Low Risk 🟢" if probability >= 0.70 else "Medium Risk 🟡" if probability >= 0.65 else "Higher Risk 🟠" if probability >= 0.60 else "High Risk 🔴"

def create_smart_tips(home_team, away_team, home_xg, away_xg, total_xg, btts, clean_sheets, score_markets):
    c = []
    for line, probs in build_ou_markets(total_xg, lines=(0.5, 1.5, 2.5, 3.5, 4.5)).items():
        c.extend([{"market": f"Total Goals Over {line}", "probability": probs["Over"], "category": "Goals"}, {"market": f"Total Goals Under {line}", "probability": probs["Under"], "category": "Goals"}])
    for line, probs in build_ou_markets(home_xg, lines=(0.5, 1.5, 2.5, 3.5)).items():
        c.extend([{"market": f"{home_team} Over {line} Goals", "probability": probs["Over"], "category": "Team Goals"}, {"market": f"{home_team} Under {line} Goals", "probability": probs["Under"], "category": "Team Goals"}])
    for line, probs in build_ou_markets(away_xg, lines=(0.5, 1.5, 2.5, 3.5)).items():
        c.extend([{"market": f"{away_team} Over {line} Goals", "probability": probs["Over"], "category": "Team Goals"}, {"market": f"{away_team} Under {line} Goals", "probability": probs["Under"], "category": "Team Goals"}])
    c.extend([{"market": "Both Teams To Score - Yes", "probability": btts["Yes"], "category": "BTTS"}, {"market": "Both Teams To Score - No", "probability": btts["No"], "category": "BTTS"}])
    c.extend([{"market": f"{home_team} Clean Sheet", "probability": clean_sheets["Home Clean Sheet"], "category": "Clean Sheet"}, {"market": f"{away_team} Clean Sheet", "probability": clean_sheets["Away Clean Sheet"], "category": "Clean Sheet"}])
    for m, p in score_markets.items(): c.append({"market": m, "probability": p, "category": "Combination"})
    
    candidates = sorted([x for x in c if x["probability"] >= 0.60], key=lambda x: x["probability"], reverse=True)
    selected, cat_count = [], {}
    for cand in candidates:
        if cat_count.get(cand["category"], 0) >= 3: continue
        cat_count[cand["category"]] = cat_count.get(cand["category"], 0) + 1
        selected.append({"market": cand["market"], "probability": round(cand["probability"], 3), "risk_level": get_risk_level(cand["probability"]), "category": cand["category"]})
        if len(selected) >= 8: break
    return selected

def extract_shap_explanation(input_df, predicted_class_index):
    try:
        fmap = {"HomeAttackVsAwayDefense": "Home Attacking Advantage vs Away Defense", "AwayAttackVsHomeDefense": "Away Attacking Advantage vs Home Defense", "GoalFormDiffL5": "Goal Difference Form Momentum", "FormDiff": "Overall Team Form Gap", "HomeGoalsMeanL5Venuedpd": "Home Team Scoring Form at Home", "AwayGoalsMeanL5Venuedpd": "Away Team Scoring Form Away", "HomeH2HL5": "Home Team Head-to-Head Record", "AwayH2HL5": "Away Team Head-to-Head Record", "MatchBalance": "Market Odds Parity", "era": "Historical Season Context"}
        vals = shap_explainer.shap_values(input_df)
        c_vals = vals[predicted_class_index] if isinstance(vals, list) else vals
        if len(c_vals.shape) > 1: c_vals = c_vals[0]
        impacts = sorted(zip(input_df.columns, c_vals), key=lambda x: x[1], reverse=True)
        return [{"factor": fmap.get(f, f), "impact_message": f"+{round(float(imp) * 100, 1)}% impact towards this outcome"} for f, imp in impacts[:3]]
    except: return [{"factor": "Model Confidence", "impact_message": "Driven by overall team form and historical stats."}]

def determine_smart_market(home_prob, draw_prob, away_prob):
    sorted_probs = sorted({"Home Win": home_prob, "Draw": draw_prob, "Away Win": away_prob}.items(), key=lambda x: x[1], reverse=True)
    top_outcome, top_prob = sorted_probs[0]
    second_outcome, second_prob = sorted_probs[1]
    gap = top_prob - second_prob
    if gap < 0.06: market_call = "Home or Draw (Double Chance)" if "Home Win" in [top_outcome, second_outcome] and "Draw" in [top_outcome, second_outcome] else "Away or Draw (Double Chance)" if "Away Win" in [top_outcome, second_outcome] and "Draw" in [top_outcome, second_outcome] else "Home or Away (No Draw / Close Call)"
    else: market_call = f"Outright {top_outcome}"
    return {"primary_prediction": top_outcome, "recommended_market": market_call, "confidence_gap": round(gap * 100, 1)}

async def process_single_prediction(match: MatchRequest):
    home_clean, away_clean = standardize_team_name(match.home_team), standardize_team_name(match.away_team)
    if home_clean not in VALID_TEAMS or away_clean not in VALID_TEAMS: raise HTTPException(status_code=400, detail=f"Unrecognized teams: {home_clean} or {away_clean}.")
    
    odds_src, odds_bookie, match_id, kickoff = "manual", None, None, None
    if None not in (match.home_odds, match.draw_odds, match.away_odds):
        home_odds, draw_odds, away_odds = float(match.home_odds), float(match.draw_odds), float(match.away_odds)
    else:
        live = await fetch_prematch_odds(home_clean, away_clean)
        if not live: raise HTTPException(status_code=400, detail=f"Could not fetch odds for {home_clean} vs {away_clean}.")
        home_odds, draw_odds, away_odds = live["home_odds"], live["draw_odds"], live["away_odds"]
        odds_src, odds_bookie, match_id, kickoff = live.get("source", "TheOddsAPI"), live.get("bookmaker"), live.get("match_id"), live.get("kickoff_utc")
    
    if home_odds <= 1 or draw_odds <= 1 or away_odds <= 1: raise HTTPException(status_code=400, detail="Invalid odds. Must be > 1.")
    
    input_df = feature_pipeline.transform(home_team=home_clean, away_team=away_clean, season=match.season, home_odds=home_odds, draw_odds=draw_odds, away_odds=away_odds)
    probs, classes = result_model.predict_proba(input_df)[0], list(result_model.classes_)
    raw_pred = result_model.predict(input_df)
    pred_val = raw_pred[0][0] if isinstance(raw_pred[0], (list, np.ndarray)) else raw_pred[0]
    p_idx, p_dict = classes.index(pred_val), {str(c): float(p) for c, p in zip(classes, probs)}
    h_prob, d_prob, a_prob = p_dict.get("H", p_dict.get("Home", 0.0)), p_dict.get("D", p_dict.get("Draw", 0.0)), p_dict.get("A", p_dict.get("Away", 0.0))
    
    h_xg, a_xg = max(0.01, float(home_goals_model.predict(input_df)[0])), max(0.01, float(away_goals_model.predict(input_df)[0]))
    t_xg = h_xg + a_xg
    
    t_markets, h_markets, a_markets = build_ou_markets(t_xg, (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)), build_ou_markets(h_xg, (0.5, 1.5, 2.5, 3.5)), build_ou_markets(a_xg, (0.5, 1.5, 2.5, 3.5))
    btts, clean = calculate_btts(h_xg, a_xg), calculate_clean_sheets(h_xg, a_xg)
    s_matrix = build_score_matrix(h_xg, a_xg, 6)
    score_markets = calculate_score_markets(s_matrix, home_clean, away_clean)
    
    g_tips = [{"market": f"Total Match Goals Over {l}", "probability": p["Over"], "risk_level": get_risk_level(p["Over"])} for l, p in t_markets.items()] + \
             [{"market": f"Total Match Goals Under {l}", "probability": p["Under"], "risk_level": get_risk_level(p["Under"])} for l, p in t_markets.items()] + \
             [{"market": f"{home_clean} Goals Over {l}", "probability": p["Over"], "risk_level": get_risk_level(p["Over"])} for l, p in h_markets.items()] + \
             [{"market": f"{away_clean} Goals Over {l}", "probability": p["Over"], "risk_level": get_risk_level(p["Over"])} for l, p in a_markets.items()] + \
             [{"market": "Both Teams To Score - Yes", "probability": btts["Yes"], "risk_level": get_risk_level(btts["Yes"])}, {"market": "Both Teams To Score - No", "probability": btts["No"], "risk_level": get_risk_level(btts["No"])}]
    
    smart_tips = sorted([x for x in g_tips if x["probability"] >= 0.60], key=lambda x: x["probability"], reverse=True)[:5]
    if not smart_tips: smart_tips = [{"market": "Skip Goal Markets", "probability": 0.0, "risk_level": "High Risk 🔴"}]

    return {
        "match": f"{home_clean} vs {away_clean}", "match_id": match_id, "kickoff_utc": kickoff,
        "odds": {"source": odds_src, "bookmaker": odds_bookie, "home": home_odds, "draw": draw_odds, "away": away_odds},
        "winner": str(pred_val), "market_analysis": determine_smart_market(h_prob, d_prob, a_prob),
        "probabilities": {"HomeWin": round(h_prob, 4), "Draw": round(d_prob, 4), "AwayWin": round(a_prob, 4)},
        "expected_goals": {"home": round(h_xg, 2), "away": round(a_xg, 2), "total": round(t_xg, 2)},
        "goal_markets": {"total_match_goals": t_markets, "home_team_goals": h_markets, "away_team_goals": a_markets},
        "both_teams_to_score": btts, "clean_sheet": clean,
        "double_chance": calculate_double_chance(h_prob, d_prob, a_prob), "draw_no_bet": calculate_draw_no_bet(h_prob, d_prob, a_prob),
        "combination_markets": score_markets, "correct_score": get_correct_scores(s_matrix, 5),
        "smart_goal_tip": smart_tips, "smart_betting_tips": create_smart_tips(home_clean, away_clean, h_xg, a_xg, t_xg, btts, clean, score_markets),
        "explanation": extract_shap_explanation(input_df, p_idx)
    }

@app.get("/")
def health_check(): return {"status": "online", "message": "BetCO Engine is running.", "odds_provider": "TheOddsAPI", "engine_version": "10.0.0", "markets_supported": ["1X2", "Double Chance", "Draw No Bet", "Total Goals", "Home Team Goals", "Away Team Goals", "BTTS", "Clean Sheet", "Correct Score", "Win + Goals", "Win + BTTS", "Win To Nil", "Winning Margin"]}

@app.get("/debug-odds")
async def debug_odds(home_team: str = "Arsenal", away_team: str = "Coventry", api_key: str = Depends(get_api_key)):
    try: return {"success": True, "match": f"{home_team} vs {away_team}", "odds": await fetch_prematch_odds(standardize_team_name(home_team), standardize_team_name(away_team))}
    except Exception as e: return {"success": False, "match": f"{home_team} vs {away_team}", "error": str(e)}

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