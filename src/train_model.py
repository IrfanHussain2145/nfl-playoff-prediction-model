import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load processed data
data_path = os.path.join("data", "processed", "playoff_training_data.csv")
df = pd.read_csv(data_path)

# Convert TRUE/FALSE strings to booleans 
df = df.replace({'TRUE': True, 'FALSE': False})

# Optional: if any boolean columns remain, convert them to 0/1
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Drop non-feature columns that leak info or aren't predictive
drop_cols = [
    col for col in X.columns if any(sub in col for sub in [
        "Team", "Conference", "Season", "Format_7Team",
        "Playoff_Round_Reached", "Team_LostTo_Or_SBWin"
    ])
]
X = X.drop(columns=drop_cols)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, os.path.join("models", "playoff_predictor.pkl"))
print("Model saved to models/playoff_predictor.pkl")
