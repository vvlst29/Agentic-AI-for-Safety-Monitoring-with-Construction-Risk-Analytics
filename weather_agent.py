import requests
from datetime import datetime

class ConstructionWeatherAgent:
    """
    SPECIALIZED AI WEATHER AGENT
    Responsibility: Autonomous Risk Assessment for Construction Sites
    """
    def __init__(self, lat=19.0760, lon=72.8777):
        #coordinates for MUMBAI#
        
        self.url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation_probability&current_weather=true"
        self.agent_name = "Site-Weather-Intelligence-Agent"

    def analyze_site_risks(self):
        try:
            response = requests.get(self.url).json()
            curr = response['current_weather']
            
            # 1. Fetch Key Metrics
            wind = curr['windspeed']
            temp = curr['temperature']
            # Get max rain probability for the next 12 hours (Predictive Intelligence)
            rain_chance = max(response['hourly']['precipitation_probability'][:12])
            
            # 2. Construction-Specific AI Reasoning
            risk_score = 0
            reasons = []
            recommendations = []

            # CRANE SAFETY LOGIC
            if wind > 30:
                risk_score += 50
                reasons.append(f"High Wind detected ({wind} km/h)")
                recommendations.append("HALT CRANE OPERATIONS: Wind exceeds safety threshold.")
            
            # CONCRETE & FOUNDATION LOGIC
            if rain_chance > 40:
                risk_score += 40
                reasons.append(f"High Rain Forecast ({rain_chance}%)")
                recommendations.append("DELAY CONCRETE POURING: Rain will compromise structural integrity.")

            # WORKER HEALTH LOGIC
            if temp < 0 or temp > 38:
                risk_score += 20
                reasons.append(f"Extreme Temperature ({temp}°C)")
                recommendations.append("MANDATORY BREAKS: Implement heat/cold stress protocol for labor.")

            # 3. Determine Final Risk Level
            status = "LOW"
            if risk_score >= 70: status = "CRITICAL"
            elif risk_score >= 30: status = "MODERATE"

            return {
                "status": status,
                "score": risk_score,
                "temp": temp,
                "wind": wind,
                "rain_prob": rain_chance,
                "findings": reasons,
                "actions": recommendations
            }
        except Exception as e:
            return {"status": "ERROR", "findings": [f"Connection failed: {e}"]}

    def display_report(self):
        data = self.analyze_site_risks()
        
        print("\n" + "="*50)
        print(f"       AI WEATHER INTELLIGENCE REPORT")
        print(f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*50)
        print(f"OVERALL RISK LEVEL: {data['status']} ({data['score']}/100)")
        print("-" * 50)
        print(f"METRICS:")
        print(f" - Temperature:    {data['temp']}°C")
        print(f" - Wind Speed:     {data['wind']} km/h")
        print(f" - Rain Forecast:  {data['rain_prob']}% (Next 6hrs)")
        
        if data['findings']:
            print("\nAI FINDINGS:")
            for item in data['findings']:
                print(f" ⚠ {item}")
            
            print("\nREQUIRED ACTIONS:")
            for action in data['actions']:
                print(f" [!] {action}")
        else:
            print("\nRESULT: Site conditions are optimal for all construction activities.")
        print("="*50 + "\n")

# --- EXECUTION ---
if __name__ == "__main__":
    # Initialize the Agent
    agent = ConstructionWeatherAgent()
    # Run the Intelligence Report
    agent.display_report()