# src/data_format.py

import pandas as pd

REQUIRED_COLUMNS = [
    "Season", "Team", "Conference", "Seed", "Record_Wins", "Div_Winner",
    "Div_Record_Wins", "Sharp_OL_Rank", "Turnover_Diff", "Def_Sacks",
    "Off_Pts_Scored", "Def_Pts_Allowed", "Pt_Differential",
    "Time_Of_Possession_Rank", "Last5_Wins", "AllPro_Count",
    "Format_7Team", "Playoff_Round_Reached"
]

def load_feature_data(path: str) -> pd.DataFrame:
    print(f"\n📂 Loading: {path}")
    df = pd.read_csv(path)

    # Validate columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing columns: {missing}")
    print("✅ All required columns present.")

    # Convert logical types
    df["Div_Winner"] = df["Div_Winner"].astype(bool)
    df["Format_7Team"] = df["Format_7Team"].astype(bool)

    # Check for missing values
    nulls = df.isnull().sum()
    if nulls.any():
        print("⚠️ Missing values:")
        print(nulls[nulls > 0])
    else:
        print("✅ No missing values.")

    print(f"📊 Loaded {len(df)} rows.\n")
    return df
