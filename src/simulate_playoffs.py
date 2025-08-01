import pandas as pd
import sys
import os
import joblib
from predict_matchup import build_matchup_row

# --- Load year from command-line ---
if len(sys.argv) < 2:
    print("Usage: python3 src/simulate_playoffs.py <YEAR>")
    sys.exit(1)

season = int(sys.argv[1])
print(f"\n🏆 Simulating {season} NFL Playoffs\n")

# --- Load data ---
matchup_path = os.path.join("data", "processed", f"matchups_{season}.csv")
stats_path = os.path.join("data", "processed", "playoff_team_features.csv")
model_path = os.path.join("models", "playoff_predictor.pkl")

matchups_df = pd.read_csv(matchup_path)
stats_df = pd.read_csv(stats_path)
model = joblib.load(model_path)

# --- Clean boolean columns ---
stats_df = stats_df.replace({"TRUE": True, "FALSE": False})
bool_cols = stats_df.select_dtypes(include='bool').columns
stats_df[bool_cols] = stats_df[bool_cols].astype(int)

# --- Wild Card Round ---
wc_games = matchups_df[matchups_df["Round"] == 1]
all_results = []
winners_by_conf = {"AFC": [], "NFC": []}

print("🏟 Wild Card Round - AFC")
for _, row in wc_games.iterrows():
    team1, team2 = row["Team_A"], row["Team_B"]
    conf = stats_df[(stats_df["Season"] == season) & (stats_df["Team"] == team1)]["Conference"].values[0]

    if conf == "AFC":
        X, _, _ = build_matchup_row(team1, team2, season, stats_df)
        X = X[model.feature_names_in_]

        proba = model.predict_proba(X)[0]
        pred = model.classes_[proba.argmax()]
        confidence = proba.max()

        winner = team1 if pred == 1 else team2
        print(f"🏈 {season} {team1} vs {season} {team2} → Winner: {season} {pred} (Confidence: {confidence:.2f})")
        winners_by_conf[conf].append(winner)
        all_results.append((season, 1, team1, team2, winner))

print("\n🏟 Wild Card Round - NFC")
for _, row in wc_games.iterrows():
    team1, team2 = row["Team_A"], row["Team_B"]
    conf = stats_df[(stats_df["Season"] == season) & (stats_df["Team"] == team1)]["Conference"].values[0]

    if conf == "NFC":
        X, _, _ = build_matchup_row(team1, team2, season, stats_df)
        X = X[model.feature_names_in_]
        
        proba = model.predict_proba(X)[0]
        pred = model.classes_[proba.argmax()]
        confidence = proba.max()

        winner = team1 if pred == 1 else team2
        print(f"🏈 {season} {team1} vs {season} {team2} → Winner: {season} {pred} (Confidence: {confidence:.2f})")
        winners_by_conf[conf].append(winner)
        all_results.append((season, 1, team1, team2, winner))

# --- Helper to get seed ---
def get_seed(team):
    return stats_df[
        (stats_df["Season"] == season) & (stats_df["Team"] == team)
    ]["Seed"].values[0]

# --- Simulate Divisional Round ---
def simulate_divisional_round(conf, wc_winners):
    print(f"\n🏟 Divisional Round - {conf}")

    # Add #1 seed to wild card winners
    top_seed_team = stats_df[
        (stats_df["Season"] == season) & 
        (stats_df["Seed"] == 1) & 
        (stats_df["Conference"] == conf)
    ]["Team"].values[0]

    teams = wc_winners + [top_seed_team]

    # Sort teams by seed (lowest number = highest rank)
    teams = sorted(teams, key=get_seed)

    winners = []
    while len(teams) >= 2:
        # Pair highest seed with lowest
        team1 = teams[0]
        team2 = teams[-1]
        teams = teams[1:-1]  # Remove used teams

        X, _, _ = build_matchup_row(team1, team2, season, stats_df)
        X = X[model.feature_names_in_]

        proba = model.predict_proba(X)[0]
        pred = model.classes_[proba.argmax()]
        confidence = proba.max()

        winner = team1 if pred == 1 else team2

        print(f"🏈 {season} {team1} vs {season} {team2} → Winner: {season} {pred} (Confidence: {confidence:.2f})")
        winners.append(winner)
        all_results.append((season, 2, team1, team2, winner))

    if teams:
        print(f"⚠️  Odd number of teams in {conf}, dropping last team: {teams[0]}")

    return winners

# Simulate both conferences
afc_div_winners = simulate_divisional_round("AFC", winners_by_conf["AFC"])
nfc_div_winners = simulate_divisional_round("NFC", winners_by_conf["NFC"])

# --- Simulate Conference Championship ---
def simulate_conference_championship(conf, div_winners):
    print(f"\n🏟 Conference Championship - {conf}")
    if len(div_winners) != 2:
        print(f"❌ Error: Expected 2 teams in {conf} championship, got {len(div_winners)}")
        return None

    team1, team2 = sorted(div_winners, key=get_seed)
    X, _, _ = build_matchup_row(team1, team2, season, stats_df)
    X = X[model.feature_names_in_]
    
    proba = model.predict_proba(X)[0]
    pred = model.classes_[proba.argmax()]
    confidence = proba.max()

    winner = team1 if pred == 1 else team2
    print(f"🏈 {season} {team1} vs {season} {team2} → Winner: {season} {pred} (Confidence: {confidence:.2f})")
    all_results.append((season, 3, team1, team2, winner))
    return winner

# Simulate each conference final
afc_champion = simulate_conference_championship("AFC", afc_div_winners)
nfc_champion = simulate_conference_championship("NFC", nfc_div_winners)

# --- Simulate Super Bowl ---
if afc_champion and nfc_champion:
    print(f"\n🏟 Super Bowl")
    team1, team2 = sorted([afc_champion, nfc_champion], key=get_seed)
    X, _, _ = build_matchup_row(team1, team2, season, stats_df)
    X = X[model.feature_names_in_]

    proba = model.predict_proba(X)[0]
    pred = model.classes_[proba.argmax()]
    confidence = proba.max()

    winner = team1 if pred == 1 else team2
    print(f"🏈 {season} {team1} vs {season} {team2} → Winner: {season} {pred} (Confidence: {confidence:.2f})")
    all_results.append((season, 4, team1, team2, winner))
    print(f"\n👑 Predicted Super Bowl Champion: {season} {winner}")
else:
    print("\n❌ Error: Could not simulate Super Bowl due to missing conference champions.")

# --- Save Results ---
results_df = pd.DataFrame(all_results, columns=["Season", "Round", "Team_A", "Team_B", "Winner"])
save_path = f"data/processed/simulated_results_{season}.csv"
results_df.to_csv(save_path, index=False)
print(f"\n📝 Detailed results saved to {save_path}")
