import json
import random
from pathlib import Path
from datetime import datetime

def generate_dataset(num_samples=50):
    output_path = Path(__file__).parent / "data" / "ksem_test_cases.json"
    
    scenarios = ["A", "B", "C"]
    data = []
    
    for i in range(num_samples):
        # Ensure at least one of each, then random
        if i < len(scenarios):
            scenario_type = scenarios[i]
        else:
            scenario_type = random.choice(scenarios)
            
        sample = {
            "id": f"TEST-{i+1:03d}",
            "timestamp": datetime.now().isoformat(),
            "scenario_type": scenario_type,
            "uav": {},
            "environment": {},
            "zone": {},
            "ground_truth": ""
        }
        
        # Base UAV stats
        uav_resistance = random.randint(8, 12)  # m/s
        
        if scenario_type == "A":
            # Scenario A: Semantic Risk (Wind > Resistance)
            # Not in no-fly zone
            sample["zone"]["is_no_fly"] = False
            sample["uav"]["max_wind_resistance"] = uav_resistance
            # Wind speed needs to be higher than resistance
            sample["environment"]["wind_speed"] = uav_resistance + random.uniform(1.0, 5.0)
            sample["ground_truth"] = "High Risk"
            sample["description"] = "Semantic Risk: Wind exceeds resistance"
            
        elif scenario_type == "B":
            # Scenario B: Hard Rule Violation (No-Fly Zone)
            sample["zone"]["is_no_fly"] = True
            sample["uav"]["max_wind_resistance"] = uav_resistance
            # Wind speed safe
            sample["environment"]["wind_speed"] = uav_resistance - random.uniform(1.0, 3.0)
            sample["ground_truth"] = "High Risk"
            sample["description"] = "Hard Rule: No-fly zone violation"
            
        elif scenario_type == "C":
            # Scenario C: Safe
            sample["zone"]["is_no_fly"] = False
            sample["uav"]["max_wind_resistance"] = uav_resistance
            # Wind speed safe
            sample["environment"]["wind_speed"] = uav_resistance - random.uniform(1.0, 5.0)
            sample["ground_truth"] = "Low Risk"
            sample["description"] = "Safe operation"

        # Round wind speed for readability
        sample["environment"]["wind_speed"] = round(sample["environment"]["wind_speed"], 2)
        
        data.append(sample)
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"Dataset generated at {output_path.absolute()}")
    print(f"Total samples: {len(data)}")
    print(f"Scenario distribution: A={sum(1 for x in data if x['scenario_type']=='A')}, "
          f"B={sum(1 for x in data if x['scenario_type']=='B')}, "
          f"C={sum(1 for x in data if x['scenario_type']=='C')}")

if __name__ == "__main__":
    generate_dataset()

