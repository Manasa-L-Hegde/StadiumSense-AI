import os
import json
import pandas as pd
import pytest
from modules.crowd_sensing import CrowdSensingEngine

@pytest.fixture
def temp_stadium_data(tmp_path):
    """Fixture to write temporary stadium graph and telemetry files for testing."""
    graph_data = {
        "nodes": [
            {"id": "Node_A", "type": "gate", "x": 10, "y": 10, "label": "Gate A"},
            {"id": "Node_B", "type": "seating", "x": 50, "y": 50, "label": "Section B"}
        ],
        "edges": []
    }
    
    # 3 points clustered near Node A (x=10, y=10) -> cluster should form
    # 1 point near Node B (x=50, y=50) -> too isolated to form a cluster of min_samples=3
    telemetry_data = [
        {"x": 10.0, "y": 10.0, "weight": 80.0, "source": "test"},
        {"x": 10.1, "y": 9.9, "weight": 85.0, "source": "test"},
        {"x": 9.9, "y": 10.1, "weight": 75.0, "source": "test"},
        {"x": 50.0, "y": 50.0, "weight": 30.0, "source": "test"} # Noise/no cluster
    ]
    
    graph_file = tmp_path / "stadium_graph.json"
    telemetry_file = tmp_path / "crowd_telemetry.csv"
    
    with open(graph_file, "w") as f:
        json.dump(graph_data, f)
        
    pd.DataFrame(telemetry_data).to_csv(telemetry_file, index=False)
    
    return str(telemetry_file), str(graph_file)

def test_dbscan_hotspot_detection(temp_stadium_data):
    telemetry_file, graph_file = temp_stadium_data
    
    engine = CrowdSensingEngine(eps=2.0, min_samples=3)
    hotspots = engine.detect_hotspots(telemetry_path=telemetry_file, graph_path=graph_file)
    
    # Assertions
    assert len(hotspots) == 1
    hs = hotspots[0]
    assert hs["cluster_id"] == 0
    assert abs(hs["centroid_x"] - 10.0) < 0.2
    assert abs(hs["centroid_y"] - 10.0) < 0.2
    assert hs["point_count"] == 3
    assert hs["avg_density"] == 80.0
    assert hs["nearest_zone_id"] == "Node_A"
    assert hs["severity"] == "High"  # density >= 70.0 is High
    
    # Congested zones lookup check
    congested = engine.get_congested_zones()
    assert "Node_A" in congested
    assert congested["Node_A"] == "High"
    assert "Node_B" not in congested

def test_incident_simulations(tmp_path):
    from data.generate_synthetic import generate_stadium_data
    
    # Test Gate B Closed incident
    generate_stadium_data(output_dir=str(tmp_path), incident="Gate B Closed")
    
    telem_file = tmp_path / "crowd_telemetry.csv"
    assert os.path.exists(telem_file)
    
    df = pd.read_csv(telem_file)
    # Check that we have a high density center near Gate B (x=90, y=50)
    gate_b_points = df[(df["x"] > 80) & (df["x"] < 100) & (df["y"] > 40) & (df["y"] < 60)]
    assert len(gate_b_points) >= 40
    assert gate_b_points["weight"].mean() > 80.0
