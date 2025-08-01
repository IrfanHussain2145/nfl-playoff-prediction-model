import pandas as pd
import os

# --- Load team features ---
features_path = os.path.join("data", "processed", "playoff_team_features.csv")
features_df = pd.read_csv(features_path)

# --- Combine all matchup files ---
matchup_dir = os.path.join("data", "processed")
matchup_files = [f for f in os.listdir(matchup_dir) if f.startswith("matchups_") and f.endswith(".csv")]

matchup_dfs = []
for f in matchup_files:
    df = pd.read_csv(os.path.join(matchup_dir, f))
    df["Season"] = int(f.split("_")[1].split(".")[0])
    matchup_dfs.append(df)

matchups_all = pd.concat(matchup_dfs, ignore_index=True)

# --- Build training rows ---
rows = []
for _, row in matchups_all.iterrows():
    season, team_a, team_b, winner = row["Season"], row["Team_A"], row["Team_B"], row["Winner"]
    fa = features_df[(features_df["Season"] == season) & (features_df["Team"] == team_a)]
    fb = features_df[(features_df["Season"] == season) & (features_df["Team"] == team_b)]
    
    if fa.empty or fb.empty:
        continue

    # Prefix A/B columns to differentiate
    row_data = {
        f"A_{col}": fa[col].values[0] for col in fa.columns if col not in ["Team"]
    }
    row_data.update({
        f"B_{col}": fb[col].values[0] for col in fb.columns if col not in ["Team"]
    })
    
    # Outcome label
    row_data["Label"] = 1 if winner == team_a else 0
    rows.append(row_data)

# --- Save training data ---
training_df = pd.DataFrame(rows)
save_path = os.path.join("data", "processed", "playoff_training_data.csv")
training_df.to_csv(save_path, index=False)
print(f"✅ Training data saved to {save_path} ({len(training_df)} rows)")
