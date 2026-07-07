import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

class CrowdSensingEngine:
    """
    Clustering engine to detect crowd hotspots in the stadium using DBSCAN on telemetry.
    Maps cluster centroids to nearest stadium nodes and evaluates severity.
    """
    def __init__(self, eps=5.0, min_samples=3):
        self.eps = eps
        self.min_samples = min_samples
        self.hotspots = []
        
    def detect_hotspots(self, telemetry_path="data/crowd_telemetry.csv", graph_path="data/stadium_graph.json"):
        """
        Runs DBSCAN clustering on the x, y coordinates from telemetry data.
        Returns a list of hotspot dictionaries.
        """
        if not os.path.exists(telemetry_path) or not os.path.exists(graph_path):
            return []
            
        # Load telemetry and graph
        df = pd.read_csv(telemetry_path)
        with open(graph_path, "r") as f:
            graph_data = json.load(f)
            
        nodes = graph_data["nodes"]
        
        if df.empty or not nodes:
            return []
            
        # Run DBSCAN on x, y coordinates
        coords = df[["x", "y"]].values
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean')
        df["cluster"] = db.fit_predict(coords)
        
        self.hotspots = []
        unique_clusters = sorted([c for c in df["cluster"].unique() if c >= 0])
        
        for cid in unique_clusters:
            c_pts = df[df["cluster"] == cid]
            centroid_x = float(c_pts["x"].mean())
            centroid_y = float(c_pts["y"].mean())
            point_count = len(c_pts)
            avg_density = float(c_pts["weight"].mean())
            
            # Map centroid to nearest stadium zone/node
            min_dist = float('inf')
            nearest_node = None
            
            for node in nodes:
                dist = np.sqrt((node["x"] - centroid_x)**2 + (node["y"] - centroid_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node
            
            # Classify severity
            if point_count >= 15 or avg_density >= 70.0:
                severity = "High"
            elif point_count >= 5 or avg_density >= 40.0:
                severity = "Medium"
            else:
                severity = "Low"
                
            self.hotspots.append({
                "cluster_id": int(cid),
                "centroid_x": round(centroid_x, 2),
                "centroid_y": round(centroid_y, 2),
                "point_count": point_count,
                "avg_density": round(avg_density, 1),
                "nearest_zone_id": nearest_node["id"] if nearest_node else "Unknown",
                "nearest_zone_label": nearest_node["label"] if nearest_node else "Unknown",
                "distance_to_zone": round(float(min_dist), 2),
                "severity": severity
            })
            
        return self.hotspots

    def get_congested_zones(self):
        """
        Helper that returns a mapping of zone_id -> severity for easy lookup.
        """
        congested = {}
        for hs in self.hotspots:
            zone_id = hs["nearest_zone_id"]
            severity = hs["severity"]
            # Keep the highest severity if multiple clusters map to same zone
            if zone_id not in congested:
                congested[zone_id] = severity
            else:
                current_sev = congested[zone_id]
                if severity == "High" or (severity == "Medium" and current_sev == "Low"):
                    congested[zone_id] = severity
        return congested
