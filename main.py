import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

class MemecoinTracker:
    def __init__(self):
        self.api_url = "https://api.dexscreener.io/latest/dex/pairs"
        self.historical_data = pd.DataFrame()
        self.model = RandomForestClassifier()

    def fetch_data(self):
        """Fetch live data from Dexscreener."""
        response = requests.get(self.api_url)
        if response.status_code == 200:
            data = response.json()
            pairs = data.get("pairs", [])
            return pd.DataFrame(pairs)
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return pd.DataFrame()

    def preprocess_data(self, df):
        """Preprocess data for modeling."""
        df = df[["priceUsd", "volumeUsd24h", "liquidityUsd"]]
        df.dropna(inplace=True)
        df = df.astype(float)
        df["price_change"] = df["priceUsd"].pct_change()
        df = df.fillna(0)
        return df

    def train_model(self, historical_df):
        """Train a machine learning model on historical data."""
        X = historical_df[["volumeUsd24h", "liquidityUsd", "price_change"]]
        y = (historical_df["price_change"] > 0.5).astype(int)  # Label as 1 if price > 50%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        print(f"Model Accuracy: {accuracy_score(y_test, predictions)}")

    def predict_opportunities(self, live_df):
        """Predict which coins are about to 100x."""
        X_live = live_df[["volumeUsd24h", "liquidityUsd", "price_change"]]
        predictions = self.model.predict(X_live)
        live_df["prediction"] = predictions
        opportunities = live_df[live_df["prediction"] == 1]
        return opportunities

    def run(self):
        """Run the tracker."""
        while True:
            print("Fetching live data...")
            live_data = self.fetch_data()
            if live_data.empty:
                print("No data fetched, retrying...")
                time.sleep(60)
                continue

            live_data = self.preprocess_data(live_data)
            print("Analyzing opportunities...")
            opportunities = self.predict_opportunities(live_data)

            print(f"Found {len(opportunities)} potential opportunities.")
            print(opportunities[["priceUsd", "volumeUsd24h", "liquidityUsd"]])
            time.sleep(300)  # Run every 5 minutes


if __name__ == "__main__":
    tracker = MemecoinTracker()
    historical = tracker.fetch_data()
    processed_historical = tracker.preprocess_data(historical)
    tracker.train_model(processed_historical)
    tracker.run()
