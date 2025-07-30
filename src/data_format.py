# src/data_format.py

import pandas as pd

EXPECTED_COLUMNS = [
    "Season", "Team", "Conference", "Seed",
    "Record_Wins", "Record_Losses",
    "DivWinner", "DivRecord_Wins", "DivRecord_Losses",
    "Sharp_OL_Rank", "Sharp_Turnover_Rank", "Sharp_SacksPressures_Rank",
    "Sharp_Off_Rank", "Sharp_Def_Rank", "TimeOfPossession",
    "PointDifferential", "Last5_Wins", "Last5_Losses",
    "AllPro_Count", "Format_7Team", "PlayoffWins"
]

def load_feature_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Validate columns
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Optional: convert types
    df['DivWinner'] = df['DivWinner'].astype(bool)
    df['Format_7Team'] = df['Format_7Team'].astype(bool)

    return df
