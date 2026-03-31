import json
import random
from pathlib import Path
from datetime import datetime

def generate_large_scale_dataset(num_samples=1000):
    output_path = Path(__file__).parent / "data" / "ksem_large_dataset.json"
    
    scenarios = ["A", "B", "C"]
    data = []
    
    for i in range(num_samples):
        # Balanced sampling
        scenario_type = scenarios[i % len(scenarios)]
            
        sample = {
            "id": f"TEST-{i+1:04d}",
            "timestamp": datetime.now().isoformat(),
            "scenario_type": scenario_type,
            "uav": {},
            "environment": {},
            "zone": {},
            "ground_truth": ""
        }
        
        # Base UAV stats with Gaussian fuzzing
        # Mean resistance 10 m/s, std dev 1.5
        uav_resistance = random.gauss(10.0, 1.5)
        uav_resistance = round(max(5.0, min(15.0, uav_resistance)), 2) # Clamp to [5, 15]
        
        # Battery fuzzing (0-100%)
        battery_level = random.gauss(60.0, 20.0)
        battery_level = round(max(0.0, min(100.0, battery_level)), 1)
        
        sample["uav"]["max_wind_resistance"] = uav_resistance
        sample["uav"]["battery"] = battery_level
        
        if scenario_type == "A":
            # Scenario A: Semantic Risk (Wind > Resistance)
            # Not in no-fly zone
            sample["zone"]["is_no_fly"] = False
            
            # Fuzz wind speed: resistance + (0.1 to 5.0)
            # Introduce edge cases near boundary
            margin = random.choice([random.uniform(0.1, 0.5), random.uniform(1.0, 5.0)])
            wind_speed = uav_resistance + margin
            
            sample["ground_truth"] = "High Risk"
            sample["description"] = f"Semantic Risk: Wind ({wind_speed:.2f}m/s) exceeds resistance ({uav_resistance}m/s)"
            
        elif scenario_type == "B":
            # Scenario B: Hard Rule Violation (No-Fly Zone)
            sample["zone"]["is_no_fly"] = True
            
            # Wind speed safe (resistance - margin)
            margin = random.uniform(1.0, 5.0)
            wind_speed = max(0.0, uav_resistance - margin)
            
            sample["ground_truth"] = "High Risk"
            sample["description"] = "Hard Rule: No-fly zone violation"
            
        elif scenario_type == "C":
            # Scenario C: Safe
            sample["zone"]["is_no_fly"] = False
            
            # Wind speed safe
            # Include edge cases (very close to limit but safe)
            is_edge_case = random.random() < 0.2 # 20% chance of edge case
            if is_edge_case:
                # Safe margin 0.1 to 0.5 m/s
                margin = random.uniform(0.1, 0.5)
            else:
                # Normal margin
                margin = random.uniform(2.0, 8.0)
                
            wind_speed = max(0.0, uav_resistance - margin)
            
            sample["ground_truth"] = "Low Risk"
            sample["description"] = "Safe operation"

        sample["environment"]["wind_speed"] = round(wind_speed, 2)
        
        data.append(sample)
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"Dataset generated at {output_path.absolute()}")
    print(f"Total samples: {len(data)}")
    
    count_a = sum(1 for x in data if x['scenario_type']=='A')
    count_b = sum(1 for x in data if x['scenario_type']=='B')
    count_c = sum(1 for x in data if x['scenario_type']=='C')
    
    print(f"Scenario distribution:")
    print(f"  A (Semantic Risk): {count_a} ({count_a/len(data):.1%})")
    print(f"  B (Hard Rule):     {count_b} ({count_b/len(data):.1%})")
    print(f"  C (Safe):          {count_c} ({count_c/len(data):.1%})")

if __name__ == "__main__":
    generate_large_scale_dataset()

