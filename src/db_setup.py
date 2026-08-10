import sqlite3
import pandas as pd
from pathlib import Path
def initialize_database():
    # 1. Load your raw CSV files
    BASE_DIR = Path(__file__).resolve().parent.parent

    df1_raw = pd.read_csv(BASE_DIR /'data'/ "df1_raw.csv")
    team_df_raw = pd.read_csv(BASE_DIR /'data'/ "team_df_raw.csv")

    # 2. Connect to SQLite (creates the file if it doesn't exist)
    conn = sqlite3.connect('football_data.db')

    # 3. Store DataFrames as SQL tables
    df1_raw.to_sql('matches_raw', conn, if_exists='replace', index=False)
    team_df_raw.to_sql('team_stats_raw', conn, if_exists='replace', index=False)

    print("Database successfully initialized with existing CSV data!")
    conn.close()

if __name__ == "__main__":
    initialize_database()