import sqlite3
import pandas as pd
import numpy as np
import difflib
import argparse
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "football_data.db"
DATA_DIR = BASE_DIR / "data"

# ==========================================
# 1. TEAM STANDARDIZATION LOGIC (UPDATED FOR LA LIGA)
# ==========================================
VALID_TEAMS = [
    # EPL Teams
    'Man City', 'West Ham', 'Middlesbrough', 'Southampton', 'Everton',
    'Aston Villa', 'Bradford', 'Arsenal', 'Ipswich', 'Newcastle',
    'Liverpool', 'Chelsea', 'Man United', 'Tottenham', 'Charlton',
    'Sunderland', 'Derby', 'Coventry', 'Leicester', 'Leeds',
    'Blackburn', 'Bolton', 'Fulham', 'West Brom', 'Birmingham',
    'Wolves', 'Portsmouth', 'Crystal Palace', 'Norwich', 'Wigan',
    'Watford', 'Sheffield United', 'Reading', 'Stoke', 'Hull',
    'Burnley', 'Blackpool', 'Swansea', 'QPR', 'Cardiff', 'Bournemouth',
    'Huddersfield', 'Brighton', 'Brentford', "Nott'm Forest", 'Luton',
    
    # La Liga Teams
    'Barcelona', 'Sociedad', 'Betis', 'Zaragoza', 'Real Madrid',
    'Malaga', 'Getafe', 'La Coruna', 'Villarreal', 'Santander',
    'Osasuna', 'Ath Madrid', 'Celta', 'Mallorca', 'Ath Bilbao',
    'Espanol', 'Cadiz', 'Alaves', 'Sevilla', 'Valencia', 'Gimnastic',
    'Levante', 'Recreativo', 'Valladolid', 'Almeria', 'Murcia',
    'Numancia', 'Sp Gijon', 'Tenerife', 'Xerez', 'Hercules',
    'Vallecano', 'Granada', 'Elche', 'Cordoba', 'Eibar', 'Las Palmas',
    'Leganes', 'Girona', 'Huesca', 'Oviedo'
]

TEAM_ALIASES = {
    # EPL Aliases
    "manchester united": "Man United", "man utd": "Man United",
    "manchester city": "Man City", "nottingham forest": "Nott'm Forest",
    "spurs": "Tottenham", "wolverhampton": "Wolves",
    "wolverhampton wanderers": "Wolves", "newcastle united": "Newcastle",
    "west ham united": "West Ham", "leeds united": "Leeds",
    "leicester city": "Leicester", "queens park rangers": "QPR",
    
    # La Liga Aliases
    "atletico madrid": "Ath Madrid", "atlético madrid": "Ath Madrid",
    "real sociedad": "Sociedad", "real betis": "Betis",
    "deportivo la coruña": "La Coruna", "deportivo la coruna": "La Coruna",
    "racing santander": "Santander", "espanyol": "Espanol",
    "sporting gijon": "Sp Gijon", "sporting gijón": "Sp Gijon",
    "rayo vallecano": "Vallecano", "athletic bilbao": "Ath Bilbao",
    "celta vigo": "Celta",
    "deportivo alavés": "Alaves", "deportivo alaves": "Alaves",
    "deportivo": "La Coruna",
}

def standardize_team_name(input_name: str) -> str:
    if not isinstance(input_name, str): return input_name
    clean_input = input_name.strip().lower()
    if clean_input in TEAM_ALIASES: return TEAM_ALIASES[clean_input]
    for valid_team in VALID_TEAMS:
        if clean_input == valid_team.lower(): return valid_team
    matches = difflib.get_close_matches(clean_input, [t.lower() for t in VALID_TEAMS], n=1, cutoff=0.6)
    if matches:
        for valid_team in VALID_TEAMS:
            if valid_team.lower() == matches[0]: return valid_team
    return input_name.strip().title()

# ==========================================
# 2. FEATURE ENGINEERING HELPERS
# ==========================================
def season_to_era(season):
    """Handles both EPL '2024/2025' and La Liga '506' formats"""
    s = str(season).strip()
    if "/" in s:
        start_year = int(s.split("/")[0])
    else:
        if len(s) == 3: start_year = 2000 + int(s[0])
        elif len(s) == 4: start_year = 2000 + int(s[:2])
        else: return 0 
    return 2026 - start_year

def calc_win_streak(results):
    streak, out = 0, []
    for result in results:
        out.append(streak)
        if result == 'W': streak += 1
        elif result in ['L', 'D']: streak = 0
    return out

def get_ho(df): return 'W' if df['FullTimeResult']=='H' else ('D' if df['FullTimeResult']=='D' else 'L')
def get_aw(df): return 'W' if df['FullTimeResult']=='A' else ('D' if df['FullTimeResult']=='D' else 'L')
def get_ho_points(df): return 3 if df['FullTimeResult']=='H' else (1 if df['FullTimeResult']=='D' else 0)
def get_aw_points(df): return 3 if df['FullTimeResult']=='A' else (1 if df['FullTimeResult']=='D' else 0)
def get_points(df):
    if (df['Venue']=='Home' and df['Result']=='H') or (df['Venue']=='Away' and df['Result']=='A'): return 3
    elif df['Result']=='D': return 1
    else: return 0
    
def create_rolling_feature(df, source, column, window):
    df = df.sort_values('MatchDate').copy()
    return df.groupby(source)[column].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())

# ==========================================
# 3. MAIN DATA PIPELINE
# ==========================================
def update_training_data(league_name="EPL"):
    print(f"\n🚀 Updating features for {league_name}...")
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Dynamic Table Selection based on league
    matches_table = "matches_raw" if league_name.upper() == "EPL" else f"matches_raw_{league_name.lower()}"
    team_table = "team_stats_raw" if league_name.upper() == "EPL" else f"team_stats_raw_{league_name.lower()}"

    print(f"Fetching raw data from {matches_table} and {team_table}...")
    print(f"Fetching raw data from {matches_table} and {team_table}...")
    try:
        # 1. Fetch data without SQL ORDER BY to avoid column name crashes
        df1 = pd.read_sql_query(f"SELECT * FROM {matches_table}", conn)
        team_df = pd.read_sql_query(f"SELECT * FROM {team_table}", conn)
        
        # 2. Unify the date columns (Legacy CSVs used 'Date', new ingest uses 'MatchDate')
        if 'Date' in df1.columns and 'MatchDate' not in df1.columns:
            df1 = df1.rename(columns={'Date': 'MatchDate'})
        if 'Date' in team_df.columns and 'MatchDate' not in team_df.columns:
            team_df = team_df.rename(columns={'Date': 'MatchDate'})
            
        # 3. Safely convert to datetime and sort chronologically in Pandas
        # (Handling mixed European date formats just in case!)
        df1['MatchDate'] = pd.to_datetime(df1['MatchDate'], dayfirst=True)
        df1 = df1.sort_values('MatchDate')
        
        team_df['MatchDate'] = pd.to_datetime(team_df['MatchDate'], dayfirst=True)
        team_df = team_df.sort_values(['Team', 'MatchDate'])
        
    except sqlite3.OperationalError as e:
        print(f"❌ Database error (tables might not exist): {e}")
        conn.close()
        return

    conn.close()

    print("Standardizing team names...")
    df1['HomeTeam'] = df1['HomeTeam'].apply(standardize_team_name)
    df1['AwayTeam'] = df1['AwayTeam'].apply(standardize_team_name)
    team_df['Team'] = team_df['Team'].apply(standardize_team_name)
    team_df['Opponent'] = team_df['Opponent'].apply(standardize_team_name)

    print("Running feature engineering logic...")

    # team_df overall rollings
    team_df["GoalsMeanL5"] = create_rolling_feature(team_df, 'Team', 'GoalsFor', 5)
    team_df["GoalsConcededL5"] = create_rolling_feature(team_df, 'Team', 'GoalsAgainst', 5)
    team_df['ShotsOTMeanL5'] = create_rolling_feature(team_df, 'Team', 'Shots', 5)
    team_df['points'] = team_df.apply(get_points, axis=1)
    team_df["PointsMeanL5"] = create_rolling_feature(team_df, 'Team', 'points', 5)
    team_df['goal_df'] = team_df.GoalsFor - team_df.GoalsAgainst
    team_df['GoalDFMeanL5'] = create_rolling_feature(team_df, 'Team', 'goal_df', 5)
    team_df['RedCardL2'] = create_rolling_feature(team_df, 'Team', 'RedCard', 2)
    team_df['GoalDFMeanL10'] = create_rolling_feature(team_df, 'Team', 'goal_df', 10)
    team_df['ShotsOTMeanL10'] = create_rolling_feature(team_df, 'Team', 'Shots', 10)
    team_df['PointsMeanL10'] = create_rolling_feature(team_df, 'Team', 'points', 10)
    team_df['GoalsMeanL10'] = create_rolling_feature(team_df, 'Team', 'GoalsFor', 10)
    team_df['GoalsConcededL10'] = create_rolling_feature(team_df, 'Team', 'GoalsAgainst', 10)
    
    result_map = {0: 'L', 1: 'D', 3: 'W'}
    team_df['outcome'] = team_df['points'].map(result_map)
    team_df['win_streak'] = team_df.groupby('Team')['outcome'].transform(calc_win_streak)
    team_df['MatchDate'] = pd.to_datetime(team_df['MatchDate'])
    team_df = team_df.sort_values(['Team', 'MatchDate']).reset_index(drop=True)
    team_df['H2HL5'] = create_rolling_feature(team_df, ['Team', 'Opponent'], 'points', 5)
    
    # df1 venue-dependent rollings
    # CRITICAL FIX: Use the new season_to_era function here!
    df1['era'] = df1['Season'].apply(season_to_era) 
    
    df1['HomeGoalsMeanL5'] = create_rolling_feature(df1, 'HomeTeam', 'FullTimeHomeGoals', 5)
    df1['AwayGoalsMeanL5'] = create_rolling_feature(df1, 'HomeTeam', 'FullTimeAwayGoals', 5)
    df1['HomeGoalsConcededL5'] = create_rolling_feature(df1, 'HomeTeam', 'FullTimeAwayGoals', 5)
    df1['AwayGoalsConcededL5'] = create_rolling_feature(df1, 'HomeTeam', 'FullTimeHomeGoals', 5)
    df1['ho_goal_df'] = df1.FullTimeHomeGoals - df1.FullTimeAwayGoals
    df1['HomeGoalDFMeanL5'] = create_rolling_feature(df1, 'HomeTeam', 'ho_goal_df', 5)
    df1['aw_goal_df'] = df1.FullTimeAwayGoals - df1.FullTimeHomeGoals
    df1['AwayGoalDFMeanL5'] = round(create_rolling_feature(df1, 'AwayTeam', 'aw_goal_df', 5), 2)
    df1['ho_win_point'] = df1.apply(get_ho_points, axis=1)
    df1['aw_win_point'] = df1.apply(get_aw_points, axis=1)
    df1['HomePointsL5'] = create_rolling_feature(df1, 'HomeTeam', 'ho_win_point', 5)
    df1['AwayPointsL5'] = create_rolling_feature(df1, 'AwayTeam', 'aw_win_point', 5)
    df1.drop(columns=['ho_win_point', 'aw_win_point', 'aw_goal_df', 'ho_goal_df'], inplace=True)
    df1['HO'] = df1.apply(get_ho, axis=1)
    df1['AW'] = df1.apply(get_aw, axis=1)
    df1['WinStreakL5Home'] = df1.groupby('HomeTeam')['HO'].transform(calc_win_streak)
    df1['WinStreakL5Away'] = df1.groupby('AwayTeam')['AW'].transform(calc_win_streak)
        
    df1.rename(columns={
        'HomeGoalsMeanL5': 'HomeGoalsMeanL5Venuedpd',
        'AwayGoalsMeanL5': 'AwayGoalsMeanL5Venuedpd',
        'HomeGoalsConcededL5': 'HomeGoalsConcededL5Venuedpd',
        'AwayGoalsConcededL5': 'AwayGoalsConcededL5Venuedpd',
        'HomeGoalDFMeanL5': 'HomeGoalDFMeanL5Venuedpd',
        'AwayGoalDFMeanL5': 'AwayGoalDFMeanL5Venuedpd',
        'HomePointsL5': 'HomePointsL5Venuedpd',
        'AwayPointsL5': 'AwayPointsL5Venuedpd'
    }, inplace=True)

    print("Merging venue and overall features...")
    overall_cols = [
        'GoalsMeanL5', 'GoalsConcededL5', 'ShotsOTMeanL10', 
        'GoalDFMeanL10', 'GoalDFMeanL5', 'PointsMeanL5', 'PointsMeanL10', 
        'GoalsMeanL10', 'GoalsConcededL10', 'win_streak', 'H2HL5', 'RedCardL2'
    ]
    
    home_overall = team_df[['MatchDate', 'Team'] + overall_cols].copy()
    home_overall.columns = ['MatchDate', 'HomeTeam'] + ['Home' + col for col in overall_cols]
    
    away_overall = team_df[['MatchDate', 'Team'] + overall_cols].copy()
    away_overall.columns = ['MatchDate', 'AwayTeam'] + ['Away' + col for col in overall_cols]

    df1['MatchDate'] = pd.to_datetime(df1['MatchDate'])
    
    merged_df = pd.merge(df1, home_overall, on=['MatchDate', 'HomeTeam'], how='inner')
    merged_df = pd.merge(merged_df, away_overall, on=['MatchDate', 'AwayTeam'], how='inner')

    print("Calculating market and derived features...")
    merged_df['HomeProb'] = (1 / merged_df['HomeOdds']) / ((1 / merged_df['HomeOdds']) + (1 / merged_df['DrawOdds']) + (1 / merged_df['AwayOdds']))
    merged_df['DrawProb'] = (1 / merged_df['DrawOdds']) / ((1 / merged_df['HomeOdds']) + (1 / merged_df['DrawOdds']) + (1 / merged_df['AwayOdds']))
    merged_df['AwayProb'] = (1 / merged_df['AwayOdds']) / ((1 / merged_df['HomeOdds']) + (1 / merged_df['DrawOdds']) + (1 / merged_df['AwayOdds']))
    merged_df['MatchBalance'] = abs(merged_df['HomeProb'] - merged_df['AwayProb'])

    merged_df['HomeAttackVsAwayDefense'] = merged_df['HomeGoalsMeanL5'] - merged_df['AwayGoalsConcededL5']
    merged_df['AwayAttackVsHomeDefense'] = merged_df['AwayGoalsMeanL5'] - merged_df['HomeGoalsConcededL5']
    merged_df['GoalFormDiffL5'] = merged_df['HomeGoalDFMeanL5'] - merged_df['AwayGoalDFMeanL5']
    merged_df['FormDiff'] = merged_df['HomePointsMeanL5'] - merged_df['AwayPointsMeanL5']

    print("Cleaning up final dataset...")
    target_cols = ['FullTimeResult', 'FullTimeHomeGoals', 'FullTimeAwayGoals']
    feature_cols = [
        'HomeTeam', 'AwayTeam', 'era', 'HomeGoalsMeanL5Venuedpd',
        'AwayGoalsMeanL5Venuedpd', 'HomeGoalsConcededL5Venuedpd',
        'AwayGoalsConcededL5Venuedpd', 'HomeGoalDFMeanL5Venuedpd',
        'AwayGoalDFMeanL5Venuedpd', 'HomePointsL5Venuedpd',
        'AwayPointsL5Venuedpd', 'WinStreakL5Home', 'WinStreakL5Away',
        'HomeOdds', 'DrawOdds', 'AwayOdds', 'HomeProb', 'DrawProb', 'AwayProb',
        'MatchBalance', 'HomeGoalsConcededL5', 'HomeGoalsMeanL5',
        'HomeShotsOTMeanL10', 'HomeGoalDFMeanL10', 'HomeGoalDFMeanL5',
        'HomePointsMeanL5', 'HomePointsMeanL10', 'HomeGoalsMeanL10',
        'HomeGoalsConcededL10', 'Homewin_streak', 'HomeH2HL5', 'HomeRedCardL2',
        'AwayGoalsConcededL5', 'AwayGoalsMeanL5', 'AwayShotsOTMeanL10',
        'AwayGoalDFMeanL10', 'AwayGoalDFMeanL5', 'AwayPointsMeanL5',
        'AwayPointsMeanL10', 'AwayGoalsMeanL10', 'AwayGoalsConcededL10',
        'Awaywin_streak', 'AwayH2HL5', 'AwayRedCardL2',
        'HomeAttackVsAwayDefense', 'AwayAttackVsHomeDefense', 'GoalFormDiffL5',
        'FormDiff'
    ]

    merged_df = merged_df.dropna(subset=feature_cols)

    X_train = merged_df[feature_cols]
    y_train = merged_df[target_cols]

    # 4. Save to League-Specific Files
    out_x = DATA_DIR / f"X_train_{league_name.lower()}.csv"
    out_y = DATA_DIR / f"y_train_{league_name.lower()}.csv"
    
    print(f"Saving updated {out_x.name} and {out_y.name}...")
    X_train.to_csv(out_x, index=False)
    y_train.to_csv(out_y, index=False)
    
    print(f"✅ Success! {league_name} X_train shape: {X_train.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=str, default="EPL", help="League to process (EPL, LaLiga, etc.)")
    args = parser.parse_args()
    
    update_training_data(args.league)