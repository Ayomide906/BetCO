import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "football_data.db"
DATA_DIR = BASE_DIR / "data"
def calc_win_streak(results):
    streak=0
    out=[]
    for result in results:
        out.append(streak)
        if result=='W':
            streak+=1
        elif result=='L' or result=='D':
            streak=0
    return out
def get_ho(df):
    if df['FullTimeResult']=='D':
        ho='D'
    elif df['FullTimeResult']=='H':
        ho='W'
    else:
        ho='L'
    return ho
def get_aw(df):
    if df['FullTimeResult']=='D':
        aw='D'
    elif df['FullTimeResult']=='H':
        aw='L'
    else:
        aw='W'
    return aw
def get_ho_points(df):
    if df['FullTimeResult']=='D':
        ho=1
    elif df['FullTimeResult']=='H':
        ho=3
    else:
        ho=0
    return ho
def get_aw_points(df):
    if df['FullTimeResult']=='D':
        aw=1
    elif df['FullTimeResult']=='H':
        aw=0
    else:
        aw=3
    return aw
def calc_win_streak(results):
    streak=0
    out=[]
    for result in results:
        out.append(streak)
        if result=='W':
            streak+=1
        elif result=='L' or result=='D':
            streak=0
    return out
def get_points(df):
    if (df['Venue']=='Home' and df['Result']=='H') or (df['Venue']=='Away' and df['Result']=='A'):
        point=3
    elif df['Result']=='D':
        point=1
    else:
        point=0
    return point
    
def create_rolling_feature(df,source,column,window):
    df=df.sort_values('MatchDate').copy()
    feature=df.groupby(source)[column].transform(
        lambda s:s.shift(1).rolling(window,min_periods=1).mean()
    )
    return feature

def update_training_data():
    print("Fetching raw data from database...")
    conn = sqlite3.connect(DB_PATH)
    
    # Load data and ensure strict chronological order
    df1 = pd.read_sql_query("SELECT * FROM matches_raw ORDER BY MatchDate", conn)
    team_df = pd.read_sql_query("SELECT * FROM team_stats_raw ORDER BY Team, MatchDate", conn)
    conn.close()

    print("Running feature engineering logic...")

    # =====================================================================
    # ⬇️ PASTE ALL YOUR NOTEBOOK FEATURE ENGINEERING CODE HERE ⬇️
    # =====================================================================
    # Example (replace with your actual notebook code):
    
    # 1. team_df overall rollings
    team_df["GoalsMeanL5"] = create_rolling_feature(team_df,'Team','GoalsFor',5)
    team_df["GoalsConcededL5"] = create_rolling_feature(team_df,'Team','GoalsAgainst',5)
    team_df['ShotsOTMeanL5'] = create_rolling_feature(team_df,'Team','Shots',5)
    team_df['points']=team_df.apply(get_points,axis=1)
    team_df["PointsMeanL5"] = create_rolling_feature(team_df,'Team','points',5)
    team_df['goal_df']=team_df.GoalsFor - team_df.GoalsAgainst
    team_df['GoalDFMeanL5']=create_rolling_feature(team_df,'Team','goal_df',5)
    team_df['RedCardL2']= create_rolling_feature(team_df,'Team','RedCard',2)
    team_df['GoalDFMeanL10']=create_rolling_feature(team_df,'Team','goal_df',10)
    team_df['ShotsOTMeanL10']=create_rolling_feature(team_df,'Team','Shots',10)
    team_df['PointsMeanL10']=create_rolling_feature(team_df,'Team','points',10)
    team_df['GoalsMeanL10']=create_rolling_feature(team_df,'Team','GoalsFor',10)
    team_df['GoalsConcededL10']=create_rolling_feature(team_df,'Team','GoalsAgainst',10)
    result_map={
        0:'L',
        1:'D',
        3:'W'
        }
    team_df['outcome']=team_df['points'].map(result_map)
    team_df['win_streak']=team_df.groupby('Team')['outcome'].transform(calc_win_streak)
    team_df['MatchDate']=pd.to_datetime(team_df['MatchDate'])
    team_df = (
        team_df
        .sort_values(['Team', 'MatchDate'])
        .reset_index(drop=True)
        )
    team_df['H2HL5']=create_rolling_feature(team_df,['Team','Opponent'],'points',5)
    # 2. df1 venue-dependent rollings
    df1['era']=df1['Season'].apply(lambda x: 2026 -int(x.split('/')[0]))
    df1['HomeGoalsMeanL5']=create_rolling_feature(df1,'HomeTeam','FullTimeHomeGoals',5)
    df1['AwayGoalsMeanL5']=create_rolling_feature(df1,'HomeTeam','FullTimeAwayGoals',5)
    df1['HomeGoalsConcededL5']=create_rolling_feature(df1,'HomeTeam','FullTimeAwayGoals',5)
    df1['AwayGoalsConcededL5']=create_rolling_feature(df1,'HomeTeam','FullTimeHomeGoals',5)
    df1['ho_goal_df']=df1.FullTimeHomeGoals - df1.FullTimeAwayGoals
    df1['HomeGoalDFMeanL5']=create_rolling_feature(df1,'HomeTeam','ho_goal_df',5)
    df1['aw_goal_df']=df1.FullTimeAwayGoals - df1.FullTimeHomeGoals
    df1['AwayGoalDFMeanL5']=round(create_rolling_feature(df1,'AwayTeam','aw_goal_df',5),2)
    df1['ho_win_point']=df1.apply(get_ho_points,axis=1)
    df1['aw_win_point']=df1.apply(get_aw_points,axis=1)
    df1['HomePointsL5']=create_rolling_feature(df1,'HomeTeam','ho_win_point',5)
    df1['AwayPointsL5']=create_rolling_feature(df1,'AwayTeam','aw_win_point',5)
    df1.drop(columns=['ho_win_point','aw_win_point','aw_goal_df','ho_goal_df'],inplace=True)
    df1['HO']=df1.apply(get_ho,axis=1)
    df1['AW']=df1.apply(get_aw,axis=1)
    df1['WinStreakL5Home']=df1.groupby('HomeTeam')['HO'].transform(calc_win_streak)
    df1['WinStreakL5Away']=df1.groupby('AwayTeam')['AW'].transform(calc_win_streak)
        
    # ... (Your existing df1 and team_df rolling calculations) ...
    
    # Rename your df1 venue-dependent columns to match what the model expects
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

    # ==========================================
    # 3. MERGING LOGIC
    # ==========================================
    print("Merging venue and overall features...")
    
    # Isolate the overall features you calculated in team_df
    overall_cols = [
        'GoalsMeanL5', 'GoalsConcededL5', 'ShotsOTMeanL10', 
        'GoalDFMeanL10', 'GoalDFMeanL5', 'PointsMeanL5', 'PointsMeanL10', 
        'GoalsMeanL10', 'GoalsConcededL10', 'win_streak', 'H2HL5', 'RedCardL2'
    ]
    
    # Create Home prefix slice
    home_overall = team_df[['MatchDate', 'Team'] + overall_cols].copy()
    home_overall.columns = ['MatchDate', 'HomeTeam'] + ['Home' + col for col in overall_cols]
    
    # Create Away prefix slice
    away_overall = team_df[['MatchDate', 'Team'] + overall_cols].copy()
    away_overall.columns = ['MatchDate', 'AwayTeam'] + ['Away' + col for col in overall_cols]

    # Convert df1 MatchDate to datetime for safe merging
    df1['MatchDate'] = pd.to_datetime(df1['MatchDate'])
    
    # Merge them all together!
    merged_df = pd.merge(df1, home_overall, on=['MatchDate', 'HomeTeam'], how='inner')
    merged_df = pd.merge(merged_df, away_overall, on=['MatchDate', 'AwayTeam'], how='inner')

    # ==========================================
    # 4. ODDS, PROBABILITIES & DERIVED FEATURES
    # ==========================================
    print("Calculating market and derived features...")
    
    # Assuming ingest.py saved the odds as 'HomeOdds', 'DrawOdds', 'AwayOdds'
    # (If they are saved as B365H, etc., rename them here first)
    
    # Calculate implied probabilities
    merged_df['HomeProb'] = (1 / merged_df['HomeOdds']) / ((1 / merged_df['HomeOdds']) + (1 / merged_df['DrawOdds']) + (1 / merged_df['AwayOdds']))
    merged_df['DrawProb'] = (1 / merged_df['DrawOdds']) / ((1 / merged_df['HomeOdds']) + (1 / merged_df['DrawOdds']) + (1 / merged_df['AwayOdds']))
    merged_df['AwayProb'] = (1 / merged_df['AwayOdds']) / ((1 / merged_df['HomeOdds']) + (1 / merged_df['DrawOdds']) + (1 / merged_df['AwayOdds']))
    merged_df['MatchBalance'] = abs(merged_df['HomeProb'] - merged_df['AwayProb'])

    # Calculate Derived Interaction Features
    merged_df['HomeAttackVsAwayDefense'] = merged_df['HomeGoalsMeanL5'] - merged_df['AwayGoalsConcededL5']
    merged_df['AwayAttackVsHomeDefense'] = merged_df['AwayGoalsMeanL5'] - merged_df['HomeGoalsConcededL5']
    merged_df['GoalFormDiffL5'] = merged_df['HomeGoalDFMeanL5'] - merged_df['AwayGoalDFMeanL5']
    merged_df['FormDiff'] = merged_df['HomePointsMeanL5'] - merged_df['AwayPointsMeanL5']

    # =====================================================================
    # ⬆️ END OF NOTEBOOK CODE ⬆️
    # =====================================================================
    # =====================================================================
    # ⬆️ END OF NOTEBOOK CODE ⬆️
    # =====================================================================

    print("Cleaning up final dataset...")

    # Define the exact 48 features your model expects, plus the target label
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

    # Drop any rows with NaN values (which usually happen in the first 5-10 matches of a season due to rolling windows)
    merged_df = merged_df.dropna(subset=feature_cols)

    # Slice the dataframe to keep ONLY the required columns (this auto-drops raw columns)
    X_train = merged_df[feature_cols]
    y_train = merged_df[target_cols]

    print("Saving updated X_train.csv and y_train.csv...")
    X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
    y_train.to_csv(DATA_DIR / "y_train.csv", index=False)
    
    print(f"Success! X_train shape: {X_train.shape}")

if __name__ == "__main__":
    update_training_data()