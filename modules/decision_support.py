import os
import json
import numpy as np
import google.generativeai as genai

class DecisionSupportSystem:
    """
    Synthesizes crowd telemetry and navigation bottlenecks into actionable alerts for stadium stewards.
    Uses rule-based logic first, then calls Gemini to refine the alert copy.
    """
    def __init__(self, graph_path="data/stadium_graph.json"):
        self.graph_path = graph_path
        self.nodes = []
        self.load_nodes()
        
    def load_nodes(self):
        if os.path.exists(self.graph_path):
            with open(self.graph_path, "r") as f:
                graph_data = json.load(f)
                self.nodes = graph_data.get("nodes", [])
                
    def _find_closest_alternative(self, congested_node: dict, node_type: str, congested_zone_ids: set) -> dict:
        """Finds the physically closest non-congested node of the same type."""
        alternatives = [n for n in self.nodes if n["type"] == node_type and n["id"] not in congested_zone_ids and n["id"] != congested_node["id"]]
        if not alternatives:
            # Fallback to any node of same type
            alternatives = [n for n in self.nodes if n["type"] == node_type and n["id"] != congested_node["id"]]
            
        if not alternatives:
            return None
            
        # Find the one with minimum Euclidean distance
        cx, cy = congested_node["x"], congested_node["y"]
        closest = min(alternatives, key=lambda n: np.sqrt((n["x"] - cx)**2 + (n["y"] - cy)**2))
        return closest
        
    def generate_alerts(self, hotspots: list, navigation_engine=None) -> dict:
        """
        Runs rule-based logic to detect operational issues.
        Combines hotspot clusters and navigation status.
        """
        alerts = []
        congested_zone_ids = {hs["nearest_zone_id"] for hs in hotspots if hs["severity"] in ["High", "Medium"]}
        
        # 1. Hotspot Alerts (Rule-Based)
        for hs in hotspots:
            zone_id = hs["nearest_zone_id"]
            zone_label = hs["nearest_zone_label"]
            severity = hs["severity"]
            avg_density = hs["avg_density"]
            count = hs["point_count"]
            
            # Find node details
            node_detail = next((n for n in self.nodes if n["id"] == zone_id), None)
            node_type = node_detail["type"] if node_detail else "general"
            
            alt_node_str = ""
            if severity in ["High", "Medium"] and node_detail:
                alt_node = self._find_closest_alternative(node_detail, node_type, congested_zone_ids)
                if alt_node:
                    alt_node_str = f" Redirect to {alt_node['label']}."
            
            if severity == "High":
                level = "CRITICAL"
                message = f"{zone_label} is severely congested (Avg Density: {avg_density}%, Points: {count}).{alt_node_str}"
                action = f"Deploy crowd controllers to {zone_label}. Reroute pedestrian flows immediately."
            elif severity == "Medium":
                level = "WARNING"
                message = f"{zone_label} is experiencing moderate congestion (Avg Density: {avg_density}%, Points: {count}).{alt_node_str}"
                action = f"Monitor queues at {zone_label}. Place backup staff on standby."
            else:
                level = "INFO"
                message = f"{zone_label} has normal/light activity (Avg Density: {avg_density}%)."
                action = "No immediate operational action required."
                
            alerts.append({
                "level": level,
                "zone_id": zone_id,
                "zone_label": zone_label,
                "message": message,
                "suggested_action": action,
                "density": avg_density,
                "count": count
            })
            
        # 2. Navigation Congestion Checks
        # Let's check routes between main gates and seating sections to see if they are impacted
        if navigation_engine:
            gates = [n for n in self.nodes if n["type"] == "gate"]
            seatings = [n for n in self.nodes if n["type"] == "seating"]
            
            # Get congested zone lookups
            congested_map = {}
            for hs in hotspots:
                congested_map[hs["nearest_zone_id"]] = hs["severity"]
                
            route_issues_count = 0
            for gate in gates[:2]:  # Check a couple of gates to keep it efficient
                for seat in seatings[:2]:
                    # Find path without avoiding crowds
                    path_normal = navigation_engine.find_shortest_path(gate["id"], seat["id"], avoid_crowds=False)
                    # Find path avoiding crowds
                    path_avoided = navigation_engine.find_shortest_path(gate["id"], seat["id"], congested_zones=congested_map, avoid_crowds=True)
                    
                    if "error" not in path_normal and "error" not in path_avoided:
                        # If the path changed, it means normal route is blocked/congested!
                        if path_normal["path"] != path_avoided["path"]:
                            route_issues_count += 1
                            
            if route_issues_count > 0:
                alerts.append({
                    "level": "WARNING",
                    "zone_id": "Navigation",
                    "zone_label": "Pedestrian Walkways",
                    "message": f"Alternative routing active: {route_issues_count} primary gate-to-seat walkways are congested and detoured.",
                    "suggested_action": "Update digital stadium wayfinding signage to trigger detour routes.",
                    "density": 50.0,
                    "count": route_issues_count
                })
                
        return {"raw_alerts": alerts}
        
    def format_alerts_genai(self, alerts_data: dict) -> str:
        """
        Uses Gemini API to convert raw rule-based alerts into a polished, plain-English radio/dispatch brief.
        Falls back to rule-based formatting if Gemini is unavailable.
        """
        raw_alerts = alerts_data.get("raw_alerts", [])
        if not raw_alerts:
            return "✅ Stadium operations normal. No active alerts."
            
        # Format the rule-based output as fallback in the structured operations format
        fallback_lines = []
        for a in raw_alerts:
            fallback_lines.append(
                f"**Situation**\n"
                f"{a['message']}\n"
                f"↓\n"
                f"**Recommended Action**\n"
                f"{a['suggested_action']}\n"
                f"↓\n"
                f"**Reason**\n"
                f"Telemetry reports {a['count']} active crowd markers with avg density at {a['density']}%.\n"
                f"↓\n"
                f"**Priority**\n"
                f"{a['level']}\n"
                f"↓\n"
                f"**Expected Impact**\n"
                f"Reduce local bottlenecks and balance pedestrian distribution."
            )
        fallback_text = "\n\n---\n\n".join(fallback_lines)
        
        # Try Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return fallback_text
            
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            raw_alerts_summary = ""
            for idx, a in enumerate(raw_alerts):
                raw_alerts_summary += f"Alert #{idx+1}: [{a['level']}] {a['message']} | Suggested Action: {a['suggested_action']}\n"
                
            prompt = (
                f"You are the Stadium Operations Director for the FIFA World Cup 2026.\n"
                f"Below is a list of raw rule-based alerts triggered by our sensor telemetry:\n\n"
                f"{raw_alerts_summary}\n"
                f"Please synthesize these warnings into a single cohesive dispatch briefing. "
                f"You MUST use EXACTLY the following structured format for each major operational alert (use the exact headers and the arrow symbol '↓' on its own line in between):\n\n"
                f"Situation\n"
                f"[Summary of the current operational issue or bottleneck]\n"
                f"↓\n"
                f"Recommended Action\n"
                f"[Tactical directives for stewards, e.g. dispatching controllers, opening gates, or rerouting]\n"
                f"↓\n"
                f"Reason\n"
                f"[Why this directive is critical, citing density figures or navigation detours]\n"
                f"↓\n"
                f"Priority\n"
                f"[CRITICAL / WARNING / INFO]\n"
                f"↓\n"
                f"Expected Impact\n"
                f"[Goal achieved by the action, e.g. reducing gate queue or routing traffic cleanly]\n\n"
                f"Keep the entire briefing under 180 words. Do not write any intro, outro, or conversational remarks."
            )
            
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            return fallback_text
        except Exception:
            return fallback_text
