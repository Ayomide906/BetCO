import sqlite3
import pandas as pd
import difflib
from pathlib import Path
from datetime import datetime

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "football_data.db"

# Automatically calculate the football-data season string (e.g., '2627')
now = datetime.now()
if now.month >= 8:
    season = f"{str(now.year)[-2:]}{str(now.year + 1)[-2:]}"
else:
    season = f"{str(now.year - 1)[-2:]}{str(now.year)[-2:]}"

# Define our leagues and their specific routing details
LEAGUES = {
    "EPL": {
        "url": f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv",
        "matches_table": "matches_raw",
        "team_table": "team_stats_raw"
    },
    "LaLiga": {
        "url": f"https://www.football-data.co.uk/mmz4281/{season}/SP1.csv",
        "matches_table": "matches_raw_laliga",
        "team_table": "team_stats_raw_laliga"
    }
}

# ==========================================
# 2. TEAM STANDARDIZATION LOGIC
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
    
    # La Liga Teams (from your unique list)
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
    "spurs": "Tottenham", "tottenham hotspur": "Tottenham",
    "wolverhampton": "Wolves", "wolverhampton wanderers": "Wolves",
    "newcastle united": "Newcastle", "west ham united": "West Ham",
    "leeds united": "Leeds", "leicester city": "Leicester",
    "queens park rangers": "QPR", "coventry city": "Coventry",
    "hull city": "Hull", "ipswich town": "Ipswich",
    "brighton & hove albion": "Brighton", "brighton and hove albion": "Brighton",
    "aston villa": "Aston Villa", "crystal palace": "Crystal Palace",
    "charlton athletic": "Charlton", "bolton wanderers": "Bolton",
    "blackburn rovers": "Blackburn", "sheffield united": "Sheffield United",
    "west bromwich albion": "West Brom", "west brom": "West Brom",
    "bristol city": "Bristol City", "luton town": "Luton",
    "brentford fc": "Brentford", "bournemouth": "Bournemouth",
    "huddersfield town": "Huddersfield",
    
    # La Liga Aliases
    "atletico madrid": "Ath Madrid", "atlético madrid": "Ath Madrid",
    "real sociedad": "Sociedad",
    "real betis": "Betis",
    "deportivo la coruña": "La Coruna", "deportivo la coruna": "La Coruna",
    "racing santander": "Santander",
    "espanyol": "Espanol",
    "sporting gijon": "Sp Gijon", "sporting gijón": "Sp Gijon",
    "rayo vallecano": "Vallecano",
    "athletic bilbao": "Ath Bilbao",
    "celta vigo": "Celta"
}

def standardize_team_name(input_name: str) -> str:
    if not isinstance(input_name, str):
        return str(input_name)
    clean_input = input_name.strip().lower()
    tags_to_remove = [" f.c.", " a.f.c.", " football club", " fc", " afc"]
    for tag in tags_to_remove:
        if clean_input.endswith(tag):
            clean_input = clean_input[:-len(tag)].strip()
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
# 3. INGESTION PIPELINE
# ==========================================
def fetch_and_ingest():
    conn = sqlite3.connect(DB_PATH)

    for league_name, config in LEAGUES.items():
        print(f"\n--- Processing {league_name} ---")
        try:
            print(f"Fetching latest matches from {config['url']}...")
            latest_matches = pd.read_csv(config['url'])
        except Exception as e:
            print(f"Warning: Could not fetch {league_name} data ({e}). Skipping...")
            continue

        # Standardize date format
        latest_matches['Date'] = pd.to_datetime(latest_matches['Date'], dayfirst=True).dt.strftime('%Y-%m-%d')

        print("Standardizing team names...")
        latest_matches['HomeTeam'] = latest_matches['HomeTeam'].apply(standardize_team_name)
        latest_matches['AwayTeam'] = latest_matches['AwayTeam'].apply(standardize_team_name)

        # Check existing data in the specific league table
        try:
            existing_df = pd.read_sql_query(f"SELECT MatchDate, HomeTeam, AwayTeam FROM {config['matches_table']}", conn)
        except sqlite3.OperationalError:
            # Table might not exist yet, create empty DataFrame
            existing_df = pd.DataFrame(columns=['MatchDate', 'HomeTeam', 'AwayTeam'])

        merged = latest_matches.merge(
            existing_df,
            left_on=['Date', 'HomeTeam', 'AwayTeam'],
            right_on=['MatchDate', 'HomeTeam', 'AwayTeam'],
            how='left',
            indicator=True
        )
        new_matches = merged[merged['_merge'] == 'left_only'].copy()

        if new_matches.empty:
            print(f"No new matches found for {league_name}. Database is up to date.")
            continue

        print(f"Found {len(new_matches)} new matches to insert for {league_name}.")

        # Format matches_raw (df1)
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
            'HomeOdds': new_matches.get('B365H', pd.NA),
            'DrawOdds': new_matches.get('B365D', pd.NA),
            'AwayOdds': new_matches.get('B365A', pd.NA)
        })

        # Format team_stats_raw (team_df)
        new_home = pd.DataFrame({
            'MatchDate': new_matches['Date'], 'Season': '2024/2025',
            'Team': new_matches['HomeTeam'], 'Opponent': new_matches['AwayTeam'],
            'Venue': 'Home', 'Shots': new_matches['HST'],
            'GoalsFor': new_matches['FTHG'], 'GoalsAgainst': new_matches['FTAG'],
            'Result': new_matches['FTR'], 'RedCard': new_matches['HR']
        })

        new_away = pd.DataFrame({
            'MatchDate': new_matches['Date'], 'Season': '2024/2025',
            'Team': new_matches['AwayTeam'], 'Opponent': new_matches['HomeTeam'],
            'Venue': 'Away', 'Shots': new_matches['AST'],
            'GoalsFor': new_matches['FTAG'], 'GoalsAgainst': new_matches['FTHG'],
            'Result': new_matches['FTR'], 'RedCard': new_matches['AR']
        })

        new_team_df = pd.concat([new_home, new_away], ignore_index=True)

        # Append to the correct league tables
        new_df1.to_sql(config['matches_table'], conn, if_exists='append', index=False)
        new_team_df.to_sql(config['team_table'], conn, if_exists='append', index=False)
        print(f"✅ {league_name} successfully updated!")

    conn.close()

if __name__ == "__main__":
    fetch_and_ingest()