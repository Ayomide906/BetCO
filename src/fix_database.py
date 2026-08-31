import sqlite3
from pathlib import Path

# Setup path exactly like your ingest.py
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "football_data.db"

def fix_seasons():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    tables = [
        "matches_raw", 
        "matches_raw_laliga", 
        "team_stats_raw", 
        "team_stats_raw_laliga"
    ]

    print("Starting database correction...")

    for table in tables:
        try:
            # Fix 2026/2027 season (Matches from August 2026 onwards)
            cursor.execute(f"""
                UPDATE {table} 
                SET Season = '2026/2027' 
                WHERE MatchDate >= '2026-08-01'
            """)
            print(f"Updated 26/27 season for {table}: {cursor.rowcount} rows fixed.")

            # Fix 2025/2026 season (Matches from August 2025 to July 2026)
            cursor.execute(f"""
                UPDATE {table} 
                SET Season = '2025/2026' 
                WHERE MatchDate >= '2025-08-01' AND MatchDate < '2026-08-01'
            """)
            print(f"Updated 25/26 season for {table}: {cursor.rowcount} rows fixed.")
            
        except sqlite3.OperationalError as e:
            print(f"Skipping {table}: {e}")

    # Commit the changes and close
    conn.commit()
    conn.close()
    print("✅ Database successfully corrected!")

if __name__ == "__main__":
    fix_seasons()