import os
import json
from functools import lru_cache
import networkx as nx
import google.generativeai as genai
from modules.security_utils import sanitize_text, validate_node_id

class NavigationEngine:
    """
    Engine to manage stadium layout graph and compute optimal routes between points.
    Integrates Gemini API to provide friendly natural language directions.
    """
    def __init__(self, graph_path="data/stadium_graph.json"):
        self.graph_path = graph_path
        self.nodes = []
        self.edges = []
        self.G = nx.Graph()
        self.load_graph()
        
    def load_graph(self):
        """Loads the stadium graph nodes and edges from json."""
        if not os.path.exists(self.graph_path):
            return
            
        with open(self.graph_path, "r") as f:
            graph_data = json.load(f)
            
        self.nodes = graph_data.get("nodes", [])
        self.edges = graph_data.get("edges", [])
        
        self.G.clear()
        # Add nodes with attributes
        for node in self.nodes:
            self.G.add_node(
                node["id"], 
                type=node["type"], 
                x=node["x"], 
                y=node["y"], 
                label=node["label"]
            )
            
        # Add edges with attributes
        for edge in self.edges:
            self.G.add_edge(
                edge["source"], 
                edge["target"], 
                distance=edge["distance"],
                base_time_seconds=edge["base_time_seconds"],
                weight=edge["distance"]  # Initial path weight is the physical distance
            )
            
    def get_node_ids(self):
        """Returns list of valid node IDs."""
        return [node["id"] for node in self.nodes]
        
    def get_node_labels_map(self):
        """Returns dict mapping node ID to friendly label."""
        return {node["id"]: node["label"] for node in self.nodes}
        
    def find_shortest_path(self, start_id: str, target_id: str, congested_zones: dict = None, avoid_crowds: bool = False):
        """
        Computes the shortest path using Dijkstra's algorithm.
        If avoid_crowds is True, penalizes edges connected to congested zones.
        """
        valid_ids = self.get_node_ids()
        try:
            start_id = validate_node_id(start_id, valid_ids)
            target_id = validate_node_id(target_id, valid_ids)
        except ValueError as e:
            return {"error": str(e)}
            
        if start_id == target_id:
            return {
                "path": [start_id],
                "distance_meters": 0.0,
                "time_seconds": 0,
                "congested_nodes_crossed": []
            }
            
        # Reset edge weights
        for u, v, d in self.G.edges(data=True):
            self.G[u][v]["weight"] = d["distance"]
            
        # Apply congestion penalties if avoid_crowds is enabled
        congested_zones = congested_zones or {}
        if avoid_crowds:
            for node_id, severity in congested_zones.items():
                if node_id in self.G:
                    # Apply penalty multiplier to all edges connected to the congested node
                    for neighbor in list(self.G.neighbors(node_id)):
                        base_dist = self.G[node_id][neighbor]["distance"]
                        if severity == "High":
                            penalty = base_dist * 100.0 + 1000.0
                        elif severity == "Medium":
                            penalty = base_dist * 10.0 + 200.0
                        else:  # Low
                            penalty = base_dist * 2.0 + 50.0
                            
                        # Keep the maximum penalty if edge is already penalized
                        self.G[node_id][neighbor]["weight"] = max(
                            self.G[node_id][neighbor]["weight"], 
                            penalty
                        )
                        
        try:
            path = nx.shortest_path(self.G, source=start_id, target=target_id, weight="weight")
        except nx.NetworkXNoPath:
            return {"error": f"No walkway route exists between {start_id} and {target_id}."}
            
        # Calculate base physical metrics (actual distance walked)
        distance = 0.0
        time_seconds = 0
        congested_crossed = []
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            distance += self.G[u][v]["distance"]
            time_seconds += self.G[u][v]["base_time_seconds"]
            
        for node_id in path:
            if node_id in congested_zones:
                congested_crossed.append((node_id, congested_zones[node_id]))
                
        return {
            "path": path,
            "distance_meters": round(distance, 1),
            "time_seconds": time_seconds,
            "congested_nodes_crossed": congested_crossed
        }
        
    def explain_route(self, path_result: dict, user_query: str = "") -> str:
        """
        Uses Google Gemini to explain the route in a natural, supportive tone.
        Falls back to rule-based English descriptions if Gemini fails or is offline.
        """
        if "error" in path_result:
            return path_result["error"]
            
        path = path_result.get("path", [])
        dist = path_result.get("distance_meters", 0.0)
        time = path_result.get("time_seconds", 0)
        congested_crossed = path_result.get("congested_nodes_crossed", [])
        
        # Clean user query for safety
        clean_query = sanitize_text(user_query, max_length=200) if user_query else "Guide me to my destination."
        
        # Immediate block of prompt injection in wayfinding
        if "[injection attempt blocked]" in clean_query:
            return "⚠️ Security Warning: System override instructions detected and blocked. Please ask a standard stadium wayfinding question."
            
        # Convert lists to tuples to make them hashable for caching
        path_tuple = tuple(path)
        congested_crossed_tuple = tuple((nid, sev) for nid, sev in congested_crossed)
        
        return self._explain_route_cached(path_tuple, dist, time, congested_crossed_tuple, clean_query)
        
    @lru_cache(maxsize=128)
    def _explain_route_cached(self, path_tuple: tuple, dist: float, time: int, congested_crossed_tuple: tuple, clean_query: str) -> str:
        labels_map = self.get_node_labels_map()
        path_labels = [labels_map.get(node_id, node_id) for node_id in path_tuple]
        
        # Format a summary of the path
        nodes_description = " -> ".join(path_labels)
        mins = time // 60
        secs = time % 60
        time_str = f"{mins} min {secs} sec" if mins > 0 else f"{secs} seconds"
        
        congested_str = ""
        if congested_crossed_tuple:
            congested_str = "Passed through congested zones: " + ", ".join([f"{labels_map.get(nid, nid)} ({sev})" for nid, sev in congested_crossed_tuple])
        else:
            congested_str = "Clear path, no heavy crowd hotspots crossed."
            
        # Rule-based fallback summary (failsafe) using the structured operations format
        fallback_desc = (
            f"**Situation**\n"
            f"Fan routing request from {labels_map.get(path_tuple[0], path_tuple[0])} to {labels_map.get(path_tuple[-1], path_tuple[-1])}.\n"
            f"↓\n"
            f"**Recommended Action**\n"
            f"Follow route: {nodes_description}\n"
            f"↓\n"
            f"**Reason**\n"
            f"Walkways calculated via graph mapping. {congested_str}\n"
            f"↓\n"
            f"**Priority**\n"
            f"Routine Operations\n"
            f"↓\n"
            f"**Expected Impact**\n"
            f"Fan reaches target covering {dist} meters in approximately {time_str}."
        )
        
        # Try to use Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return fallback_desc
            
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt = (
                f"You are a stadium routing system for the FIFA World Cup 2026.\n"
                f"A fan requested: '{clean_query}'.\n"
                f"Computed route: {nodes_description} ({dist} meters, {time_str}). Congestion: {congested_str}.\n\n"
                f"You MUST write the response in EXACTLY this structured format. Use the exact headers and arrow symbols on their own lines as shown below:\n\n"
                f"Situation\n"
                f"[One sentence explaining the routing request]\n"
                f"↓\n"
                f"Recommended Action\n"
                f"[Friendly step-by-step walking instructions specifying gates and landmarks passed]\n"
                f"↓\n"
                f"Reason\n"
                f"[Why this path is best, e.g. shortest walkway or avoids congested crowd zones]\n"
                f"↓\n"
                f"Priority\n"
                f"[Routine Operations / Special Assistance]\n"
                f"↓\n"
                f"Expected Impact\n"
                f"[Expected travel time and distance]\n\n"
                f"Keep it under 150 words total. Do not write any intro, outro, or conversational greetings."
            )
            
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            return fallback_desc
        except Exception:
            return fallback_desc
