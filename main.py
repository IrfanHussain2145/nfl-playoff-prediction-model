# main.py

from src.data_format import load_feature_data

if __name__ == "__main__":
    # Update path if your file is renamed or relocated
    csv_path = "data/processed/playoff_team_features.csv"
    
    df = load_feature_data(csv_path)
    print("\n🧪 Preview of loaded data:")
    print(df.head())
