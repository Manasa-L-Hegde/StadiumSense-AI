import os
import json
import numpy as np
import pandas as pd

def generate_stadium_data(output_dir="data", randomize=False, seed=None, incident=None):
    """
    Generates synthetic stadium graph nodes/edges and crowd density telemetry,
    saving them as json and csv files respectively.
    Also generates crowd_historical.csv to train the predictive machine learning model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Define Stadium Nodes
    nodes = [
        {"id": "Gate_A", "type": "gate", "x": 10, "y": 50, "label": "Gate A (Main West Entry)"},
        {"id": "Gate_B", "type": "gate", "x": 90, "y": 50, "label": "Gate B (East Entry)"},
        {"id": "Gate_C", "type": "gate", "x": 50, "y": 10, "label": "Gate C (North Entry)"},
        {"id": "Gate_D", "type": "gate", "x": 50, "y": 90, "label": "Gate D (South Entry)"},
        {"id": "Sec_101", "type": "seating", "x": 30, "y": 30, "label": "Section 101 (Lower West)"},
        {"id": "Sec_102", "type": "seating", "x": 30, "y": 70, "label": "Section 102 (Upper West)"},
        {"id": "Sec_103", "type": "seating", "x": 70, "y": 30, "label": "Section 103 (Lower East)"},
        {"id": "Sec_104", "type": "seating", "x": 70, "y": 70, "label": "Section 104 (Upper East)"},
        {"id": "Restroom_N", "type": "restroom", "x": 50, "y": 20, "label": "Restroom North"},
        {"id": "Restroom_S", "type": "restroom", "x": 50, "y": 80, "label": "Restroom South"},
        {"id": "Food_Stall_West", "type": "food", "x": 20, "y": 50, "label": "FIFA Fan Fest Food West"},
        {"id": "Food_Stall_East", "type": "food", "x": 80, "y": 50, "label": "Concession Corner East"},
        {"id": "Medical_West", "type": "medical", "x": 15, "y": 45, "label": "First Aid Center West"},
        {"id": "Medical_East", "type": "medical", "x": 85, "y": 45, "label": "First Aid Center East"},
        {"id": "Exit_A", "type": "exit", "x": 5, "y": 50, "label": "Main Exit West"},
        {"id": "Exit_B", "type": "exit", "x": 95, "y": 50, "label": "Main Exit East"}
    ]
    
    # 2. Define Stadium Walkway Edges
    def calc_dist(n1, n2):
        return round(float(np.sqrt((n1["x"] - n2["x"])**2 + (n1["y"] - n2["y"])**2)), 1)
    
    node_dict = {n["id"]: n for n in nodes}
    
    raw_edges = [
        ("Exit_A", "Gate_A"),
        ("Exit_B", "Gate_B"),
        ("Gate_A", "Food_Stall_West"),
        ("Gate_A", "Medical_West"),
        ("Medical_West", "Food_Stall_West"),
        ("Food_Stall_West", "Sec_101"),
        ("Food_Stall_West", "Sec_102"),
        ("Sec_101", "Gate_C"),
        ("Sec_101", "Restroom_N"),
        ("Sec_102", "Gate_D"),
        ("Sec_102", "Restroom_S"),
        ("Gate_C", "Restroom_N"),
        ("Gate_D", "Restroom_S"),
        ("Restroom_N", "Sec_103"),
        ("Restroom_S", "Sec_104"),
        ("Sec_103", "Food_Stall_East"),
        ("Sec_104", "Food_Stall_East"),
        ("Food_Stall_East", "Gate_B"),
        ("Gate_B", "Medical_East"),
        ("Medical_East", "Exit_B"),
        ("Gate_C", "Gate_D"),
        ("Sec_101", "Sec_102"),
        ("Sec_103", "Sec_104")
    ]
    
    edges = []
    for source, target in raw_edges:
        dist = calc_dist(node_dict[source], node_dict[target])
        edges.append({
            "source": source,
            "target": target,
            "distance": dist,
            "base_time_seconds": int(dist * 2)
        })
        
    graph_data = {
        "nodes": nodes,
        "edges": edges
    }
    
    with open(os.path.join(output_dir, "stadium_graph.json"), "w") as f:
        json.dump(graph_data, f, indent=4)
        
    # 3. Generate Crowd Telemetry CSV
    if seed is not None:
        np.random.seed(seed)
    elif not randomize:
        np.random.seed(42)
        
    telemetry_points = []
    
    # Noise: 40 points
    for _ in range(40):
        x = np.random.uniform(5, 95)
        y = np.random.uniform(5, 95)
        weight = np.random.uniform(10, 35)
        telemetry_points.append({
            "x": round(float(x), 2),
            "y": round(float(y), 2),
            "weight": round(float(weight), 1),
            "source": "sensor_ambient"
        })
        
    # Configure hotspots based on active incident profile
    if incident == "Gate B Closed":
        hotspot_configs = [
            {"center_x": 90, "center_y": 50, "num_pts": 50, "mean_density": 96.0, "std": 1.2, "src": "ticket_scan"}, # Gate B bottleneck
            {"center_x": 71, "center_y": 71, "num_pts": 15, "mean_density": 60.0, "std": 2.5, "src": "seat_sensor"}
        ]
    elif incident == "Concourse Concession Fire":
        hotspot_configs = [
            {"center_x": 70, "center_y": 70, "num_pts": 45, "mean_density": 95.0, "std": 1.0, "src": "camera_queue"}, # Sec 104 fire zone
            {"center_x": 12, "center_y": 51, "num_pts": 20, "mean_density": 70.0, "std": 2.0, "src": "ticket_scan"}
        ]
    elif incident == "Medical Emergency at Gate A":
        hotspot_configs = [
            {"center_x": 10, "center_y": 50, "num_pts": 35, "mean_density": 90.0, "std": 1.3, "src": "sensor_ambient"}, # Gate A emergency
            {"center_x": 49, "center_y": 21, "num_pts": 12, "mean_density": 58.0, "std": 2.0, "src": "camera_queue"}
        ]
    elif incident == "Severe Rainstorm":
        hotspot_configs = [
            {"center_x": 50, "center_y": 20, "num_pts": 30, "mean_density": 88.0, "std": 1.5, "src": "camera_queue"}, # North Restroom shelter
            {"center_x": 50, "center_y": 80, "num_pts": 30, "mean_density": 88.0, "std": 1.5, "src": "camera_queue"}, # South Restroom shelter
            {"center_x": 20, "center_y": 50, "num_pts": 35, "mean_density": 92.0, "std": 1.8, "src": "seat_sensor"}, # Food Stall West shelter
            {"center_x": 80, "center_y": 50, "num_pts": 35, "mean_density": 92.0, "std": 1.8, "src": "seat_sensor"}  # Food Stall East shelter
        ]
    else:
        # Standard Matchday configurations
        if not randomize:
            hotspot_configs = [
                {"center_x": 12, "center_y": 51, "num_pts": 30, "mean_density": 85.0, "std": 2.0, "src": "ticket_scan"},
                {"center_x": 71, "center_y": 71, "num_pts": 16, "mean_density": 65.0, "std": 3.0, "src": "seat_sensor"},
                {"center_x": 49, "center_y": 21, "num_pts": 8, "mean_density": 55.0, "std": 1.5, "src": "camera_queue"}
            ]
        else:
            all_node_ids = [n["id"] for n in nodes if n["type"] not in ["exit"]]
            chosen_nodes = np.random.choice(all_node_ids, size=3, replace=False)
            
            hotspot_configs = []
            for i, node_id in enumerate(chosen_nodes):
                node_detail = node_dict[node_id]
                num_pts = int(np.random.choice([10, 18, 28]))
                mean_density = float(np.random.choice([55.0, 72.0, 88.0]))
                src = np.random.choice(["ticket_scan", "seat_sensor", "camera_queue"])
                hotspot_configs.append({
                    "center_x": node_detail["x"],
                    "center_y": node_detail["y"],
                    "num_pts": num_pts,
                    "mean_density": mean_density,
                    "std": float(np.random.uniform(1.5, 3.5)),
                    "src": src
                })
                
    for config in hotspot_configs:
        for _ in range(config["num_pts"]):
            x = np.random.normal(config["center_x"], config["std"])
            y = np.random.normal(config["center_y"], config["std"])
            weight = np.random.normal(config["mean_density"], 8.0)
            telemetry_points.append({
                "x": round(float(np.clip(x, 0, 100)), 2),
                "y": round(float(np.clip(y, 0, 100)), 2),
                "weight": round(float(np.clip(weight, 10, 100)), 1),
                "source": config["src"]
            })
            
    df = pd.DataFrame(telemetry_points)
    df.to_csv(os.path.join(output_dir, "crowd_telemetry.csv"), index=False)
    
    # 4. Generate Historical Training Data (crowd_historical.csv) for RandomForest
    np.random.seed(100) # Seeding specifically for historical logs
    historical_records = []
    
    for node in nodes:
        node_id = node["id"]
        node_type = node["type"]
        
        # Simulate different game phases
        for time_k in range(-90, 90, 15):  # every 15 mins around kickoff
            is_halftime = 1 if (45 <= time_k <= 60) else 0
            
            # Base density logic depending on zone type and time
            if node_type == "gate":
                # Gates are busiest before kickoff (-60 to 0)
                base = 75.0 if (-45 <= time_k <= 15) else 25.0
            elif node_type == "exit":
                # Exits busy at the end (> 75)
                base = 80.0 if (time_k >= 75) else 15.0
            elif node_type == "food":
                # Food is busy halftime and before kickoff
                base = 85.0 if (is_halftime or -15 <= time_k <= 15) else 35.0
            elif node_type == "restroom":
                # Restrooms busy halftime
                base = 80.0 if is_halftime else 25.0
            elif node_type == "seating":
                # Seating busy during game (0 to 45, 60 to 90)
                base = 70.0 if (0 <= time_k <= 45 or time_k >= 60) else 20.0
            else:
                base = 20.0
                
            for _ in range(5):  # multiple observations per node and time
                current_density = float(np.clip(np.random.normal(base, 10.0), 5.0, 100.0))
                # Future density trend: simulation of simple crowd movement in 15 mins
                if node_type == "gate":
                    # Gate density goes down after kickoff
                    future_change = -15.0 if (time_k >= 0) else +10.0
                elif node_type == "exit":
                    future_change = +20.0 if (time_k >= 60) else 0.0
                elif node_type == "food" or node_type == "restroom":
                    # Spikes and drops quickly
                    future_change = -25.0 if is_halftime else (+15.0 if time_k == 30 or time_k == 75 else 0.0)
                else:
                    future_change = np.random.uniform(-5, 5)
                    
                future_density = float(np.clip(current_density + future_change + np.random.normal(0, 5.0), 0.0, 100.0))
                
                historical_records.append({
                    "zone_id": node_id,
                    "zone_type": node_type,
                    "minutes_from_kickoff": float(time_k),
                    "is_halftime": int(is_halftime),
                    "current_density": round(current_density, 1),
                    "future_density_15m": round(future_density, 1)
                })
                
    df_hist = pd.DataFrame(historical_records)
    df_hist.to_csv(os.path.join(output_dir, "crowd_historical.csv"), index=False)
    
    return len(df), len(df_hist)

if __name__ == "__main__":
    generate_stadium_data()
