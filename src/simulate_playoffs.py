import pandas as pd
import joblib
from predict_matchup import build_matchup_row

# Load trained model
model = joblib.load("models/playoff_predictor.pkl")

# Load team stats
stats_df = pd.read_csv("data/processed/playoff_team_features.csv")
stats_df = stats_df.replace({'TRUE': True, 'FALSE': False})
stats_df[stats_df.select_dtypes(include='bool').columns] = stats_df.select_dtypes(include='bool').astype(int)

# Predict winner of a game
def predict_game(team1, team2, season):
    print(f"🏈 {season} {team1} vs {season} {team2}", end=" → ")
    X, _, _ = build_matchup_row(team1, team2, season, stats_df)
    X = X[model.feature_names_in_]  # Align columns to match training
    pred = model.predict(X)[0]
    winner = team1 if pred == 1 else team2
    print(f"Winner: {season} {winner}")
    return winner

# 2022 Playoff Bracket (based on real matchups)
rounds = {
    "WC": [
        ("SF", "SEA"),
        ("MIN", "NYG"),
        ("TB", "DAL"),
        ("JAX", "LAC"),
        ("CIN", "BAL"),
        ("BUF", "MIA"),
    ],
    "Div": [],  # Will populate based on WC winners
    "Conf": [],  # Will populate based on Div winners
    "SB": [],  # Final matchup
}

# Store winners per round
results = {}

# Wild Card
print("\n🏟 Wild Card Round")
results["WC"] = [predict_game(a, b, 2022) for (a, b) in rounds["WC"]]

# Divisional Round
print("\n🏟 Divisional Round")
rounds["Div"] = [
    ("PHI", results["WC"][0]),  # PHI vs WC winner from NFC side
    (results["WC"][1], results["WC"][2]),  # Two other NFC WC winners
    ("KC", results["WC"][3]),  # KC vs WC winner from AFC side
    (results["WC"][4], results["WC"][5]),  # Two other AFC WC winners
]
results["Div"] = [predict_game(a, b, 2022) for (a, b) in rounds["Div"]]

# Conference Championships
print("\n🏟 Conference Championships")
rounds["Conf"] = [
    (results["Div"][0], results["Div"][1]),  # NFC
    (results["Div"][2], results["Div"][3]),  # AFC
]
results["Conf"] = [predict_game(a, b, 2022) for (a, b) in rounds["Conf"]]

# Super Bowl
print("\n🏆 Super Bowl")
rounds["SB"] = [(results["Conf"][0], results["Conf"][1])]
results["SB"] = [predict_game(a, b, 2022) for (a, b) in rounds["SB"]]

# Champion
print(f"\n👑 Predicted Super Bowl Champion: 2022 {results['SB'][0]}")
