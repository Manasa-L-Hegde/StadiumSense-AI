import os
import json
import pandas as pd
import pytest
from modules.crowd_forecaster import CrowdForecaster

@pytest.fixture
def temp_historical_data(tmp_path):
    # Create simple historical dataset
    hist_data = [
        {"zone_id": "Gate_A", "zone_type": "gate", "minutes_from_kickoff": -30.0, "is_halftime": 0, "current_density": 40.0, "future_density_15m": 50.0},
        {"zone_id": "Gate_A", "zone_type": "gate", "minutes_from_kickoff": -15.0, "is_halftime": 0, "current_density": 50.0, "future_density_15m": 60.0},
        {"zone_id": "Gate_A", "zone_type": "gate", "minutes_from_kickoff": 0.0, "is_halftime": 0, "current_density": 60.0, "future_density_15m": 45.0},
        {"zone_id": "Sec_101", "zone_type": "seating", "minutes_from_kickoff": 15.0, "is_halftime": 0, "current_density": 20.0, "future_density_15m": 25.0},
        {"zone_id": "Sec_101", "zone_type": "seating", "minutes_from_kickoff": 30.0, "is_halftime": 0, "current_density": 25.0, "future_density_15m": 22.0}
    ]
    
    file_path = tmp_path / "crowd_historical.csv"
    pd.DataFrame(hist_data).to_csv(file_path, index=False)
    return str(file_path)

def test_forecaster_training_and_prediction(temp_historical_data):
    forecaster = CrowdForecaster(historical_data_path=temp_historical_data)
    
    # Assertions
    assert forecaster.is_trained is True
    
    # Run prediction for known zone
    pred = forecaster.forecast_density(
        zone_id="Gate_A",
        zone_type="gate",
        current_density=45.0,
        minutes_from_kickoff=-20.0,
        is_halftime=False
    )
    
    assert isinstance(pred, float)
    assert 0.0 <= pred <= 100.0

def test_forecaster_unknown_fallback(temp_historical_data):
    forecaster = CrowdForecaster(historical_data_path=temp_historical_data)
    
    # Test fallback triggers for unknown zones
    pred = forecaster.forecast_density(
        zone_id="UNKNOWN_ZONE",
        zone_type="food",
        current_density=50.0,
        minutes_from_kickoff=0.0,
        is_halftime=False
    )
    
    # Heuristic fallback for food: change is 0 since not halftime, so it should return 50.0
    assert pred == 50.0

def test_forecaster_missing_file_fallback():
    forecaster = CrowdForecaster(historical_data_path="nonexistent_file.csv")
    assert forecaster.is_trained is False
    
    pred = forecaster.forecast_density(
        zone_id="Gate_A",
        zone_type="gate",
        current_density=40.0,
        minutes_from_kickoff=-30.0,
        is_halftime=False
    )
    
    # Heuristic for gate: change is +8.0 before kickoff, so 40.0 + 8.0 = 48.0
    assert pred == 48.0

def test_forecaster_caching(temp_historical_data):
    forecaster = CrowdForecaster(historical_data_path=temp_historical_data)
    forecaster.forecast_density.cache_clear()
    
    pred1 = forecaster.forecast_density(
        zone_id="Gate_A",
        zone_type="gate",
        current_density=45.0,
        minutes_from_kickoff=-20.0,
        is_halftime=False
    )
    
    pred2 = forecaster.forecast_density(
        zone_id="Gate_A",
        zone_type="gate",
        current_density=45.0,
        minutes_from_kickoff=-20.0,
        is_halftime=False
    )
    
    assert pred1 == pred2
    cache_info = forecaster.forecast_density.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1
