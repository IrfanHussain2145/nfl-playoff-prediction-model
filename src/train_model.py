import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load processed training data
data_path = os.path.join("data", "processed", "playoff_training_data.csv")
df = pd.read_csv(data_path)

# Convert TRUE/FALSE strings to booleans and then to 0/1
df = df.replace({'TRUE': True, 'FALSE': False})
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Shuffle data to randomize input order
df = df.sample(frac=1).reset_index(drop=True)

# Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Drop columns that leak outcome or are non-predictive
leakage_cols = [
    col for col in X.columns if any(key in col for key in [
        "Team", "Conference", "Season", "Format_7Team",
        "Playoff_Round_Reached", "Team_LostTo_Or_SBWin"
    ])
]
X = X.drop(columns=leakage_cols)

# Train/test split (fixed random_state for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model (no random_state here → allows learning variation over time)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate performance
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/playoff_predictor.pkl")
print("Model saved to models/playoff_predictor.pkl")
 