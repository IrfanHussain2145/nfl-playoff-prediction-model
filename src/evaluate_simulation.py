import pandas as pd
import os
from tabulate import tabulate
from rich import print

def load_results(season):
    sim_path = f"data/processed/simulated_results_{season}.csv"
    true_path = f"data/processed/matchups_{season}.csv"
    sim_df = pd.read_csv(sim_path)
    true_df = pd.read_csv(true_path)
    return sim_df, true_df

def evaluate_weighted(sim_df, true_df):
    total_points = 0
    max_points = 0
    mismatches = []

    # Map: (Round, frozenset({Team_A, Team_B})) → Winner
    sim_lookup = {
        (r, frozenset([a, b])): w
        for r, a, b, w in sim_df[["Round", "Team_A", "Team_B", "Winner"]].values
    }

    for _, true_row in true_df.iterrows():
        rnd = true_row["Round"]
        a, b = true_row["Team_A"], true_row["Team_B"]
        true_winner = true_row["Winner"]
        key = (rnd, frozenset([a, b]))

        max_points += 3 if rnd > 1 else 1  # Later rounds: 2 for matchup, 1 for winner

        if key in sim_lookup:
            pred_winner = sim_lookup[key]
            if rnd == 1:
                if pred_winner == true_winner:
                    total_points += 1
                else:
                    mismatches.append((rnd, a, b, true_winner, pred_winner, "Wrong Wild Card Pick"))
            else:
                total_points += 2  # correct matchup
                if pred_winner == true_winner:
                    total_points += 1
                else:
                    mismatches.append((rnd, a, b, true_winner, pred_winner, "Wrong Winner"))
        else:
            # Check if a winner for this round exists, even if matchup doesn't match
            pred_rows = sim_df[sim_df["Round"] == rnd]
            pred_winner = pred_rows["Winner"].values[0] if not pred_rows.empty else "None"
            mismatches.append((rnd, a, b, true_winner, pred_winner, "Matchup Mismatch"))

    return total_points, max_points, mismatches

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 src/evaluate_simulation.py <YEAR>")
        return

    season = int(sys.argv[1])
    sim_df, true_df = load_results(season)
    score, max_score, mismatches = evaluate_weighted(sim_df, true_df)

    print(f"\n✅ Evaluation for {season} Playoffs")
    print(f"Weighted Accuracy: {score}/{max_score} = {score / max_score:.2f}\n")

    if mismatches:
        print("❌ Mispredicted Games:")
        mismatch_df = pd.DataFrame(
            mismatches,
            columns=["Round", "Team_A", "Team_B", "Winner_True", "Winner_Pred", "Reason"]
        )
        print(tabulate(mismatch_df, headers="keys", tablefmt="psql", showindex=False))

if __name__ == "__main__":
    main()
