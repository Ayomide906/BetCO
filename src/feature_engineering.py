import pandas as pd
import numpy as np
from pathlib import Path

class LiveMatchFeatureEngineer:
    def __init__(self, df1_raw, team_df_raw):
        """
        Initialize with pristine, unshifted RAW datasets.
        """
        # Ensure strict chronological order for accurate .tail() slicing
        if "MatchDate" in df1_raw.columns:
            self.df1 = df1_raw.sort_values("MatchDate").copy()
        else:
            self.df1 = df1_raw.copy()

        if "MatchDate" in team_df_raw.columns:
            self.team_df = team_df_raw.sort_values(["Team", "MatchDate"]).copy()
        else:
            self.team_df = team_df_raw.copy()

        # Target feature list exact order
        self.feature_columns = [
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

    def fit(self, X=None, y=None):
        """
        Pre-groups data for O(1) lookups during API inference.
        """
        self.home_history = {team: group for team, group in self.df1.groupby("HomeTeam")}
        self.away_history = {team: group for team, group in self.df1.groupby("AwayTeam")}
        self.overall_history = {team: group for team, group in self.team_df.groupby("Team")}

        # H2H Lookup: grouped by (Team, Opponent) in team_df
        self.h2h_history = {
            (team, opp): group
            for (team, opp), group in self.team_df.groupby(["Team", "Opponent"])
        }
        return self

    def season_to_era(self, season):
        # Convert to string just in case it's loaded as an integer
        s = str(season).strip()
        # 1. Handle the EPL format with slashes (e.g., "2024/2025")
        if "/" in s:
            start_year = int(s.split("/")[0])
            # 2. Handle La Liga format (e.g., "506", "1011", "2425")
        else:
            if len(s) == 3:
                start_year = 2000 + int(s[0])
            elif len(s) == 4:
                start_year = 2000 + int(s[:2])
            else:
                return 0 # Fallback
        return 2026 - start_year

    def _calc_points_df1(self, df, venue):
        """Calculates points from raw df1 Result column (H, D, A)"""
        if df.empty: return 0
        if venue == 'Home':
            pts = df['FullTimeResult'].map({'H': 3, 'D': 1, 'A': 0}).sum()
        else:
            pts = df['FullTimeResult'].map({'A': 3, 'D': 1, 'H': 0}).sum()
        return pts / len(df)  # Return mean

    def _calc_points_team_df(self, df):
        """Calculates points based on GoalsFor/Against in team_df"""
        if df.empty: return 0
        conditions = [
            df['GoalsFor'] > df['GoalsAgainst'],
            df['GoalsFor'] == df['GoalsAgainst']
        ]
        pts = np.select(conditions, [3, 1], default=0)
        return pts.mean()

    def transform(self, home_team, away_team, season, home_odds, draw_odds, away_odds):
        row = {}

        # ==========================================
        # 1. IDENTIFIERS
        # ==========================================
        row["HomeTeam"] = home_team
        row["AwayTeam"] = away_team
        row["era"] = self.season_to_era(season)

        # ==========================================
        # 2. MARKET FEATURES
        # ==========================================
        row["HomeOdds"] = home_odds
        row["DrawOdds"] = draw_odds
        row["AwayOdds"] = away_odds

        inv = np.array([1 / home_odds, 1 / draw_odds, 1 / away_odds])
        probs = inv / inv.sum()
        row["HomeProb"] = probs[0]
        row["DrawProb"] = probs[1]
        row["AwayProb"] = probs[2]
        row["MatchBalance"] = abs(probs[0] - probs[2])

        # ==========================================
        # 3. VENUE-DEPENDENT FEATURES (From df1)
        # ==========================================
        # HOME TEAM AT HOME (L5)
        if home_team in self.home_history:
            h_df = self.home_history[home_team].tail(5)
            row["HomeGoalsMeanL5Venuedpd"] = h_df["FullTimeHomeGoals"].mean()
            row["HomeGoalsConcededL5Venuedpd"] = h_df["FullTimeAwayGoals"].mean()
            row["HomeGoalDFMeanL5Venuedpd"] = (h_df["FullTimeHomeGoals"] - h_df["FullTimeAwayGoals"]).mean()
            row["HomePointsL5Venuedpd"] = self._calc_points_df1(h_df, 'Home')
            row["WinStreakL5Home"] = (h_df["FullTimeResult"] == 'H').sum()
        else:
            row.update({"HomeGoalsMeanL5Venuedpd": 0, "HomeGoalsConcededL5Venuedpd": 0,
                        "HomeGoalDFMeanL5Venuedpd": 0, "HomePointsL5Venuedpd": 0, "WinStreakL5Home": 0})

        # AWAY TEAM AWAY (L5)
        if away_team in self.away_history:
            a_df = self.away_history[away_team].tail(5)
            row["AwayGoalsMeanL5Venuedpd"] = a_df["FullTimeAwayGoals"].mean()
            row["AwayGoalsConcededL5Venuedpd"] = a_df["FullTimeHomeGoals"].mean()
            row["AwayGoalDFMeanL5Venuedpd"] = (a_df["FullTimeAwayGoals"] - a_df["FullTimeHomeGoals"]).mean()
            row["AwayPointsL5Venuedpd"] = self._calc_points_df1(a_df, 'Away')
            row["WinStreakL5Away"] = (a_df["FullTimeResult"] == 'A').sum()
        else:
            row.update({"AwayGoalsMeanL5Venuedpd": 0, "AwayGoalsConcededL5Venuedpd": 0,
                        "AwayGoalDFMeanL5Venuedpd": 0, "AwayPointsL5Venuedpd": 0, "WinStreakL5Away": 0})

        # ==========================================
        # 4. OVERALL FORM FEATURES (From team_df)
        # ==========================================
        # HOME OVERALL (L5 & L10)
        if home_team in self.overall_history:
            h_all_10 = self.overall_history[home_team].tail(10)
            h_all_5 = h_all_10.tail(5)

            # L5
            row["HomeGoalsMeanL5"] = h_all_5["GoalsFor"].mean()
            row["HomeGoalsConcededL5"] = h_all_5["GoalsAgainst"].mean()
            row["HomeGoalDFMeanL5"] = (h_all_5["GoalsFor"] - h_all_5["GoalsAgainst"]).mean()
            row["HomePointsMeanL5"] = self._calc_points_team_df(h_all_5)
            row["Homewin_streak"] = (h_all_5["GoalsFor"] > h_all_5["GoalsAgainst"]).sum()

            # L10
            row["HomeGoalsMeanL10"] = h_all_10["GoalsFor"].mean()
            row["HomeGoalsConcededL10"] = h_all_10["GoalsAgainst"].mean()
            row["HomeGoalDFMeanL10"] = (h_all_10["GoalsFor"] - h_all_10["GoalsAgainst"]).mean()
            row["HomePointsMeanL10"] = self._calc_points_team_df(h_all_10)

            # Assuming 'Shots' maps to OT based on your notebook trick
            row["HomeShotsOTMeanL10"] = h_all_10.get("Shots", pd.Series(dtype=float)).mean()

            # L2
            h_all_2 = h_all_10.tail(2)
            row["HomeRedCardL2"] = h_all_2.get("RedCard", pd.Series(dtype=float)).sum()
        else:
            row.update({"HomeGoalsMeanL5": 0, "HomeGoalsConcededL5": 0, "HomeGoalDFMeanL5": 0, "HomePointsMeanL5": 0,
                        "Homewin_streak": 0, "HomeGoalsMeanL10": 0, "HomeGoalsConcededL10": 0, "HomeGoalDFMeanL10": 0,
                        "HomePointsMeanL10": 0, "HomeShotsOTMeanL10": 0, "HomeRedCardL2": 0})

        # AWAY OVERALL (L5 & L10)
        if away_team in self.overall_history:
            a_all_10 = self.overall_history[away_team].tail(10)
            a_all_5 = a_all_10.tail(5)

            # L5
            row["AwayGoalsMeanL5"] = a_all_5["GoalsFor"].mean()
            row["AwayGoalsConcededL5"] = a_all_5["GoalsAgainst"].mean()
            row["AwayGoalDFMeanL5"] = (a_all_5["GoalsFor"] - a_all_5["GoalsAgainst"]).mean()
            row["AwayPointsMeanL5"] = self._calc_points_team_df(a_all_5)
            row["Awaywin_streak"] = (a_all_5["GoalsFor"] > a_all_5["GoalsAgainst"]).sum()

            # L10
            row["AwayGoalsMeanL10"] = a_all_10["GoalsFor"].mean()
            row["AwayGoalsConcededL10"] = a_all_10["GoalsAgainst"].mean()
            row["AwayGoalDFMeanL10"] = (a_all_10["GoalsFor"] - a_all_10["GoalsAgainst"]).mean()
            row["AwayPointsMeanL10"] = self._calc_points_team_df(a_all_10)
            row["AwayShotsOTMeanL10"] = a_all_10.get("Shots", pd.Series(dtype=float)).mean()

            # L2
            a_all_2 = a_all_10.tail(2)
            row["AwayRedCardL2"] = a_all_2.get("RedCard", pd.Series(dtype=float)).sum()
        else:
            row.update({"AwayGoalsMeanL5": 0, "AwayGoalsConcededL5": 0, "AwayGoalDFMeanL5": 0, "AwayPointsMeanL5": 0,
                        "Awaywin_streak": 0, "AwayGoalsMeanL10": 0, "AwayGoalsConcededL10": 0, "AwayGoalDFMeanL10": 0,
                        "AwayPointsMeanL10": 0, "AwayShotsOTMeanL10": 0, "AwayRedCardL2": 0})

        # ==========================================
        # 5. HEAD TO HEAD (L5)
        # ==========================================
        # From Home Team perspective
        if (home_team, away_team) in self.h2h_history:
            h2h_df = self.h2h_history[(home_team, away_team)].tail(5)
            row["HomeH2HL5"] = self._calc_points_team_df(h2h_df)
        else:
            row["HomeH2HL5"] = 1.0  # Default to draw pts if no history

        # From Away Team perspective
        if (away_team, home_team) in self.h2h_history:
            h2h_df_a = self.h2h_history[(away_team, home_team)].tail(5)
            row["AwayH2HL5"] = self._calc_points_team_df(h2h_df_a)
        else:
            row["AwayH2HL5"] = 1.0

        # ==========================================
        # 6. DERIVED / INTERACTION FEATURES
        # ==========================================
        # Attack vs Defense (Using overall form L5, adjust to VenueDpd if that's what your model used)
        row["HomeAttackVsAwayDefense"] = row["HomeGoalsMeanL5"] - row["AwayGoalsConcededL5"]
        row["AwayAttackVsHomeDefense"] = row["AwayGoalsMeanL5"] - row["HomeGoalsConcededL5"]

        row["GoalFormDiffL5"] = row["HomeGoalDFMeanL5"] - row["AwayGoalDFMeanL5"]
        row["FormDiff"] = row["HomePointsMeanL5"] - row["AwayPointsMeanL5"]

        # Handle any trailing NaN values generated by zero-division or empty means
        result_df = pd.DataFrame([row]).fillna(0)

        # Guarantee exact column order for model.predict()
        return result_df[self.feature_columns]

# --- Usage Example ---
if __name__ =='__main__':
    BASE_DIR = Path(__file__).resolve().parent.parent
    df1_raw_laliga = pd.read_csv(BASE_DIR/'data'/ "df1_raw_laliga.csv")
    team_df_raw_laliga=pd.read_csv(BASE_DIR/'data'/"team_df_raw_laliga.csv")
    pipeline = LiveMatchFeatureEngineer(df1_raw_laliga, team_df_raw_laliga)
    pipeline.fit()
    sample_features = pipeline.transform(
    home_team="Barcelona",
    away_team="Betis",
    season="2025/2026",
    home_odds=1.85,
    draw_odds=3.60,
    away_odds=4.50
    )
    print(f"Feature Vector Shape: {sample_features.shape}") # Should be (1, 48)
    print("\nAny NaN values?", sample_features.isna().any().any()) # Should be False
    print(sample_features.head()[['AwayH2HL5',
                                  'HomeH2HL5']])