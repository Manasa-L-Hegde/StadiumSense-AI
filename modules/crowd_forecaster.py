import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class CrowdForecaster:
    """
    Predictive analytics engine for venue crowd density forecasting.
    Trains a Random Forest Regressor on historical time-series patterns of stadium sections
    to forecast crowd density levels 15 minutes in advance.
    """
    def __init__(self, historical_data_path="data/crowd_historical.csv"):
        self.historical_data_path = historical_data_path
        self.model = RandomForestRegressor(n_estimators=30, random_state=42)
        self.zone_id_encoder = LabelEncoder()
        self.zone_type_encoder = LabelEncoder()
        self.is_trained = False
        self.train_model()
        
    def train_model(self):
        """Loads historical data, preprocesses features, and fits the Random Forest model."""
        if not os.path.exists(self.historical_data_path):
            return
            
        try:
            df = pd.read_csv(self.historical_data_path)
            if df.empty:
                return
                
            # Fit encoders
            self.zone_id_encoder.fit(df["zone_id"])
            self.zone_type_encoder.fit(df["zone_type"])
            
            # Prepare feature matrices
            X = pd.DataFrame()
            X["zone_id_enc"] = self.zone_id_encoder.transform(df["zone_id"])
            X["zone_type_enc"] = self.zone_type_encoder.transform(df["zone_type"])
            X["minutes_from_kickoff"] = df["minutes_from_kickoff"]
            X["is_halftime"] = df["is_halftime"]
            X["current_density"] = df["current_density"]
            
            y = df["future_density_15m"]
            
            # Fit model
            self.model.fit(X, y)
            self.is_trained = True
        except Exception:
            # Silently fallback to heuristic if training fails
            self.is_trained = False
            
    def forecast_density(self, zone_id: str, zone_type: str, current_density: float, minutes_from_kickoff: float = 0.0, is_halftime: bool = False) -> float:
        """
        Predicts the density of a zone in 15 minutes using the trained RandomForest.
        Falls back to heuristic rules if the model is not trained or has encoding mismatches.
        """
        is_halftime_val = 1 if is_halftime else 0
        
        if self.is_trained:
            try:
                # Check if labels are known to the encoder, if not raise ValueError to trigger fallback
                if zone_id not in self.zone_id_encoder.classes_ or zone_type not in self.zone_type_encoder.classes_:
                    raise ValueError("Unknown label")
                    
                zone_id_enc = self.zone_id_encoder.transform([zone_id])[0]
                zone_type_enc = self.zone_type_encoder.transform([zone_type])[0]
                
                # Make prediction
                X_pred = pd.DataFrame([{
                    "zone_id_enc": zone_id_enc,
                    "zone_type_enc": zone_type_enc,
                    "minutes_from_kickoff": minutes_from_kickoff,
                    "is_halftime": is_halftime_val,
                    "current_density": current_density
                }])
                
                pred = float(self.model.predict(X_pred)[0])
                return round(np.clip(pred, 0.0, 100.0), 1)
                
            except Exception:
                pass
                
        # Heuristic Failsafe/Fallback prediction logic
        change = 0.0
        if zone_type == "gate":
            change = -10.0 if minutes_from_kickoff >= 0 else +8.0
        elif zone_type == "exit":
            change = +15.0 if minutes_from_kickoff >= 60 else 0.0
        elif zone_type in ["food", "restroom"]:
            change = -20.0 if is_halftime_val else (+10.0 if minutes_from_kickoff in [30, 75] else 0.0)
        else:
            change = float(np.random.uniform(-3, 3))
            
        future_density = current_density + change
        return round(np.clip(future_density, 0.0, 100.0), 1)
