import os
import json
import pytest
from unittest.mock import MagicMock, patch
from modules.navigation import NavigationEngine

@pytest.fixture
def temp_graph_data(tmp_path):
    # A simple triangle path: Gate A -> Sec 101 -> Gate C
    # There is also a direct walkway from Gate A -> Gate C
    # So direct path is A -> C (dist = 10)
    # Detour path is A -> S101 -> C (dist = 5 + 6 = 11)
    # If Gate C is congested, direct path to C is still taken unless we avoid it.
    # If Gate C is congested and we want to go Gate A -> Gate C: 
    # wait, if Gate C is target, we must end at C.
    # Let's make a grid:
    # A (0,0) - B (0, 10) [dist=10]
    # A (0,0) - C (10, 0) [dist=10]
    # B (0,10) - D (10, 10) [dist=10]
    # C (10,0) - D (10, 10) [dist=10]
    # Path A -> D:
    # Path 1: A -> B -> D (dist=20)
    # Path 2: A -> C -> D (dist=20)
    # If C is congested, and we avoid crowds, path should choose A -> B -> D (dist=20).
    graph_data = {
        "nodes": [
            {"id": "A", "type": "gate", "x": 0, "y": 0, "label": "Gate A"},
            {"id": "B", "type": "seating", "x": 0, "y": 10, "label": "Section B"},
            {"id": "C", "type": "seating", "x": 10, "y": 0, "label": "Section C"},
            {"id": "D", "type": "exit", "x": 10, "y": 10, "label": "Exit D"}
        ],
        "edges": [
            {"source": "A", "target": "B", "distance": 10.0, "base_time_seconds": 20},
            {"source": "A", "target": "C", "distance": 10.0, "base_time_seconds": 20},
            {"source": "B", "target": "D", "distance": 10.0, "base_time_seconds": 20},
            {"source": "C", "target": "D", "distance": 10.0, "base_time_seconds": 20}
        ]
    }
    
    file_path = tmp_path / "stadium_graph.json"
    with open(file_path, "w") as f:
        json.dump(graph_data, f)
        
    return str(file_path)

def test_shortest_path_without_congestion(temp_graph_data):
    engine = NavigationEngine(graph_path=temp_graph_data)
    
    # Path from A to D should be either [A, B, D] or [A, C, D] (dist 20)
    res = engine.find_shortest_path("A", "D")
    assert "error" not in res
    assert res["distance_meters"] == 20.0
    assert len(res["path"]) == 3
    assert res["path"][0] == "A"
    assert res["path"][-1] == "D"

def test_shortest_path_with_congestion_avoidance(temp_graph_data):
    engine = NavigationEngine(graph_path=temp_graph_data)
    
    # C is congested (High severity)
    congested = {"C": "High"}
    
    # Run with avoid_crowds=True
    res = engine.find_shortest_path("A", "D", congested_zones=congested, avoid_crowds=True)
    
    # Since C is penalized, it MUST route A -> B -> D
    assert "error" not in res
    assert res["path"] == ["A", "B", "D"]
    assert res["distance_meters"] == 20.0
    assert res["congested_nodes_crossed"] == []

def test_invalid_node_error(temp_graph_data):
    engine = NavigationEngine(graph_path=temp_graph_data)
    
    res = engine.find_shortest_path("A", "INVALID_NODE")
    assert "error" in res
    assert "does not exist in the stadium graph" in res["error"]

@patch("google.generativeai.GenerativeModel")
def test_explain_route_gemini_mock(mock_gen_model_class, temp_graph_data):
    # Mock response
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Walk from Gate A, pass through Section B, and you will reach Exit D."
    mock_model_instance.generate_content.return_value = mock_response
    mock_gen_model_class.return_value = mock_model_instance
    
    engine = NavigationEngine(graph_path=temp_graph_data)
    path_res = engine.find_shortest_path("A", "D")
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key_here"}):
        explanation = engine.explain_route(path_res, user_query="How do I get to Exit D?")
        assert "Walk from Gate A" in explanation
        mock_model_instance.generate_content.assert_called_once()
