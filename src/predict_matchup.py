import pandas as pd
import joblib
import os

def load_model():
    model_path = os.path.join("models", "playoff_predictor.pkl")
    return joblib.load(model_path)

def load_team_stats():
    stats_path = os.path.join("data", "processed", "playoff_team_features.csv")
    df = pd.read_csv(stats_path)

    df = df.replace({'TRUE': True, 'FALSE': False})
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # 👇 Convert all relevant feature columns to numeric
    numeric_cols = [
        "Seed", "Record_Wins", "Div_Winner", "Div_Record_Wins", "Sharp_PFF_OL_Rank",
        "Turnover_Diff", "Def_Sacks", "Off_Pts_Scored", "Def_Pts_Allowed",
        "Pt_Differential", "Time_Of_Possession_Rank", "Last5_Wins", "AllPro_Count"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def build_matchup_row(team1, team2, season, stats_df):
    row1 = stats_df[(stats_df["Team"] == team1) & (stats_df["Season"] == season)].iloc[0]
    row2 = stats_df[(stats_df["Team"] == team2) & (stats_df["Season"] == season)].iloc[0]

    # Include the same features used in training
    features = [
        "Seed", "Record_Wins", "Div_Winner", "Div_Record_Wins", "Sharp_PFF_OL_Rank",
        "Turnover_Diff", "Def_Sacks", "Off_Pts_Scored", "Def_Pts_Allowed",
        "Pt_Differential", "Time_Of_Possession_Rank", "Last5_Wins", "AllPro_Count"
    ]

    X = {}
    for feat in features:
        X[f"A_{feat}"] = row1[feat]
        X[f"B_{feat}"] = row2[feat]

    name1 = f"{season} {team1}"
    name2 = f"{season} {team2}"
    return pd.DataFrame([X]), name1, name2


def predict_matchup(team1, team2, season):
    print(f"🔍 Predicting {team1} vs {team2} ({season})")

    model = load_model()
    stats_df = load_team_stats()
    X, name1, name2 = build_matchup_row(team1, team2, season, stats_df)

    # Ensure feature order matches training
    X = X[model.feature_names_in_]
    pred = model.predict(X)[0]
    winner = name1 if pred == 0 else name2
    print(f"✅ Predicted winner: {winner}")

# Example call
if __name__ == "__main__":
    predict_matchup("BUF", "CIN", 2022)
