import os
import json
import pytest
from unittest.mock import MagicMock, patch
from modules.decision_support import DecisionSupportSystem

@pytest.fixture
def mock_graph_file(tmp_path):
    graph_data = {
        "nodes": [
            {"id": "Gate_A", "type": "gate", "x": 10, "y": 50, "label": "Gate A"},
            {"id": "Gate_B", "type": "gate", "x": 90, "y": 50, "label": "Gate B"},
            {"id": "Sec_101", "type": "seating", "x": 30, "y": 30, "label": "Section 101"}
        ],
        "edges": []
    }
    
    file_path = tmp_path / "stadium_graph.json"
    with open(file_path, "w") as f:
        json.dump(graph_data, f)
        
    return str(file_path)

def test_decision_support_rule_based_alerts(mock_graph_file):
    dss = DecisionSupportSystem(graph_path=mock_graph_file)
    
    # 1 High severity hotspot at Gate A
    hotspots = [
        {
            "cluster_id": 0,
            "centroid_x": 12.0,
            "centroid_y": 51.0,
            "point_count": 25,
            "avg_density": 85.0,
            "nearest_zone_id": "Gate_A",
            "nearest_zone_label": "Gate A",
            "distance_to_zone": 2.2,
            "severity": "High"
        }
    ]
    
    res = dss.generate_alerts(hotspots=hotspots)
    alerts = res["raw_alerts"]
    
    assert len(alerts) == 1
    alert = alerts[0]
    
    assert alert["level"] == "CRITICAL"
    assert alert["zone_id"] == "Gate_A"
    assert "Gate B" in alert["message"]  # Alternative gate suggestion
    assert "Deploy crowd controllers" in alert["suggested_action"]

def test_decision_support_no_hotspots(mock_graph_file):
    dss = DecisionSupportSystem(graph_path=mock_graph_file)
    res = dss.generate_alerts(hotspots=[])
    alerts = res["raw_alerts"]
    assert len(alerts) == 0

@patch("google.generativeai.GenerativeModel")
def test_decision_support_gemini_format(mock_gen_model_class, mock_graph_file):
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "STADIUM DISPATCH: Redirect fans from Gate A to Gate B immediately due to a bottleneck."
    mock_model_instance.generate_content.return_value = mock_response
    mock_gen_model_class.return_value = mock_model_instance
    
    dss = DecisionSupportSystem(graph_path=mock_graph_file)
    alerts_data = {
        "raw_alerts": [
            {
                "level": "CRITICAL",
                "zone_id": "Gate_A",
                "zone_label": "Gate A",
                "message": "Gate A is congested. Redirect to Gate B.",
                "suggested_action": "Deploy crowd controllers.",
                "density": 85.0,
                "count": 25
            }
        ]
    }
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key_here"}):
        brief = dss.format_alerts_genai(alerts_data)
        assert "STADIUM DISPATCH" in brief
        mock_model_instance.generate_content.assert_called_once()

@patch("google.generativeai.GenerativeModel")
def test_decision_support_caching(mock_gen_model_class, mock_graph_file):
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "STADIUM DISPATCH: Redirect fans from Gate A to Gate B immediately due to a bottleneck."
    mock_model_instance.generate_content.return_value = mock_response
    mock_gen_model_class.return_value = mock_model_instance
    
    dss = DecisionSupportSystem(graph_path=mock_graph_file)
    dss._format_alerts_genai_cached.cache_clear()
    
    alerts_data = {
        "raw_alerts": [
            {
                "level": "CRITICAL",
                "zone_id": "Gate_A",
                "zone_label": "Gate A",
                "message": "Gate A is congested. Redirect to Gate B.",
                "suggested_action": "Deploy crowd controllers.",
                "density": 85.0,
                "count": 25
            }
        ]
    }
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key_here"}):
        brief1 = dss.format_alerts_genai(alerts_data)
        brief2 = dss.format_alerts_genai(alerts_data)
        
        assert brief1 == brief2
        assert mock_model_instance.generate_content.call_count == 1
