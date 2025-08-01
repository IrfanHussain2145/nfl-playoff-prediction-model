import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import matplotlib.pyplot as plt

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

# Attach feature names used for later use in prediction
clf.feature_names_in_ = X.columns.tolist()

# Evaluate model
y_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, clf.predict(X_train))
print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance plot
importances = clf.feature_importances_
indices = importances.argsort()[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.close()

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, os.path.join("models", "playoff_predictor.pkl"))
print("Model saved to models/playoff_predictor.pkl")
