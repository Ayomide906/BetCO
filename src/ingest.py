import sqlite3
import pandas as pd
import difflib
from pathlib import Path

# Define paths so it works perfectly inside your src/ directory
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "football_data.db"
from datetime import datetime

# Automatically calculate the football-data season string (e.g., '2627')
now = datetime.now()
if now.month >= 8: # August or later (New season started)
    season = f"{str(now.year)[-2:]}{str(now.year + 1)[-2:]}"
else: # Jan to July (Still in the previous year's season)
    season = f"{str(now.year - 1)[-2:]}{str(now.year)[-2:]}"

# Your dynamic URL
# URL for current season EPL data from football-data.co.uk
LATEST_EPL_URL = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

# ==========================================
# 1. TEAM STANDARDIZATION LOGIC
# ==========================================
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
    "tottenham hotspur": "Tottenham",
    "wolverhampton": "Wolves",
    "wolverhampton wanderers": "Wolves",
    "newcastle united": "Newcastle",
    "west ham united": "West Ham",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "queens park rangers": "QPR",
    "coventry city": "Coventry",
    "hull city": "Hull",
    "ipswich town": "Ipswich",
    "brighton & hove albion": "Brighton",
    "brighton and hove albion": "Brighton",
    "aston villa": "Aston Villa",
    "crystal palace": "Crystal Palace",
    "charlton athletic": "Charlton",
    "bolton wanderers": "Bolton",
    "blackburn rovers": "Blackburn",
    "sheffield united": "Sheffield United",
    "west bromwich albion": "West Brom",
    "west brom": "West Brom",
    "bristol city": "Bristol City",
    "luton town": "Luton",
    "brentford fc": "Brentford",
    "bournemouth": "Bournemouth",
    "huddersfield town": "Huddersfield"
}

def standardize_team_name(input_name: str) -> str:
    if not isinstance(input_name, str):
        return str(input_name)
    
    clean_input = input_name.strip().lower()
    
    # Strip formal API/club suffixes
    tags_to_remove = [" f.c.", " a.f.c.", " football club", " fc", " afc"]
    for tag in tags_to_remove:
        if clean_input.endswith(tag):
            clean_input = clean_input[:-len(tag)].strip()
            
    # Remove leading club prefixes
    if clean_input.startswith("afc "):
        clean_input = clean_input[4:].strip()
    
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

# ==========================================
# 2. INGESTION PIPELINE
# ==========================================
def fetch_and_ingest():
    print("Fetching latest match results from football-data.co.uk...")
    
    # 1. Download current season match data
    latest_matches = pd.read_csv(LATEST_EPL_URL)

    # Standardize date format to match YYYY-MM-DD
    latest_matches['Date'] = pd.to_datetime(latest_matches['Date'], dayfirst=True).dt.strftime('%Y-%m-%d')

    # ---> CRITICAL SAFETY NET: Standardize names BEFORE checking the database <---
    print("Standardizing team names from the fetched data...")
    latest_matches['HomeTeam'] = latest_matches['HomeTeam'].apply(standardize_team_name)
    latest_matches['AwayTeam'] = latest_matches['AwayTeam'].apply(standardize_team_name)

    # Connect to local database
    conn = sqlite3.connect(DB_PATH)
    existing_df = pd.read_sql_query("SELECT MatchDate, HomeTeam, AwayTeam FROM matches_raw", conn)

    # 2. Filter out matches already stored in our database
    # Merge on date + clean team names to find net-new matches
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
        'HomeTeam': new_matches['HomeTeam'],  # These are now guaranteed clean!
        'AwayTeam': new_matches['AwayTeam'],  # These are now guaranteed clean!
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