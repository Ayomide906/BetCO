import sqlite3
import pandas as pd
from pathlib import Path

# Define paths so it works perfectly inside your src/ directory
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "football_data.db"

# URL for current season EPL data from football-data.co.uk
LATEST_EPL_URL = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"


def fetch_and_ingest():
    print("Fetching latest match results from football-data.co.uk...")
    # 1. Download current season match data
    latest_matches = pd.read_csv(LATEST_EPL_URL)

    # Standardize date format to match YYYY-MM-DD
    latest_matches['Date'] = pd.to_datetime(latest_matches['Date'], dayfirst=True).dt.strftime('%Y-%m-%d')

    # Connect to local database
    conn = sqlite3.connect(DB_PATH)
    existing_df = pd.read_sql_query("SELECT MatchDate, HomeTeam, AwayTeam FROM matches_raw", conn)

    # 2. Filter out matches already stored in our database
    # Merge on date + team names to find net-new matches
    merged = latest_matches.merge(
        existing_df,
        left_on=['Date', 'HomeTeam', 'AwayTeam'],
        right_on=['MatchDate', 'HomeTeam', 'AwayTeam'],
        how='left',
        indicator=True
    )
    new_matches = merged[merged['_merge'] == 'left_only'].copy()

    if new_matches.empty:
        print("No new matches found. Database is already up to date.")
        conn.close()
        return

    print(f"Found {len(new_matches)} new matches to insert.")

    # 3. Format new matches for `matches_raw` (df1 format) - NOW INCLUDES ODDS
    new_df1 = pd.DataFrame({
        'Season': '2024/2025',
        'MatchDate': new_matches['Date'],
        'HomeTeam': new_matches['HomeTeam'],
        'AwayTeam': new_matches['AwayTeam'],
        'FullTimeHomeGoals': new_matches['FTHG'],
        'FullTimeAwayGoals': new_matches['FTAG'],
        'FullTimeResult': new_matches['FTR'],
        'HalfTimeHomeGoals': new_matches['HTHG'],
        'HalfTimeAwayGoals': new_matches['HTAG'],
        'HalfTimeResult': new_matches['HTR'],
        'HomeShots': new_matches['HS'],
        'AwayShots': new_matches['AS'],
        'HomeShotsOnTarget': new_matches['HST'],
        'AwayShotsOnTarget': new_matches['AST'],
        'HomeCorners': new_matches['HC'],
        'AwayCorners': new_matches['AC'],
        'HomeFouls': new_matches['HF'],
        'AwayFouls': new_matches['AF'],
        'HomeYellowCards': new_matches['HY'],
        'AwayYellowCards': new_matches['AY'],
        'HomeRedCards': new_matches['HR'],
        'AwayRedCards': new_matches['AR'],
        
        # --- EXTRACTING LIVE ODDS (Bet365) ---
        # Using .get() ensures it won't crash if a column is missing for some reason
        'HomeOdds': new_matches.get('B365H', pd.NA),
        'DrawOdds': new_matches.get('B365D', pd.NA),
        'AwayOdds': new_matches.get('B365A', pd.NA)
    })

    # 4. Format new matches for `team_stats_raw` (team_df format)
    new_home = pd.DataFrame({
        'MatchDate': new_matches['Date'],
        'Season': '2024/2025',
        'Team': new_matches['HomeTeam'],
        'Opponent': new_matches['AwayTeam'],
        'Venue': 'Home',
        'Shots': new_matches['HST'],
        'GoalsFor': new_matches['FTHG'],
        'GoalsAgainst': new_matches['FTAG'],
        'Result': new_matches['FTR'],
        'RedCard': new_matches['HR']
    })

    new_away = pd.DataFrame({
        'MatchDate': new_matches['Date'],
        'Season': '2024/2025',
        'Team': new_matches['AwayTeam'],
        'Opponent': new_matches['HomeTeam'],
        'Venue': 'Away',
        'Shots': new_matches['AST'],
        'GoalsFor': new_matches['FTAG'],
        'GoalsAgainst': new_matches['FTHG'],
        'Result': new_matches['FTR'],
        'RedCard': new_matches['AR']
    })

    new_team_df = pd.concat([new_home, new_away], ignore_index=True)

    # 5. Append only new records into SQL database
    new_df1.to_sql('matches_raw', conn, if_exists='append', index=False)
    new_team_df.to_sql('team_stats_raw', conn, if_exists='append', index=False)

    print("Successfully ingested new match records with market odds!")
    conn.close()


if __name__ == "__main__":
    fetch_and_ingest()