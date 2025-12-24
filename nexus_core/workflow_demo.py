"""
SkyNet-RWA-Nexus Workflow Simulation: Auction-based Takeoff.

This script demonstrates the interaction between L2 Agents and L3 Economic Logic.
Scenario: Two UAVs compete for a congested airspace voxel.

Workflow:
1. Initialize Economic Engine (Voxel Pricing).
2. Instantiate Agents (Emergency vs Logistics).
3. Agents observe state and place bids.
4. System clears auction and simulates Smart Contract Hook.
"""

import asyncio
import random
from nexus_core.economics.pricing import CongestionPricingModel, VoxelParams
from nexus_core.mas.agents import UAVAgent, AgentType
from dataclasses import asdict

async def run_auction_simulation():
    print(">>> [SkyNet-RWA-Nexus] Initializing System...\n")
    
    # 1. Setup Economics
    econ_engine = CongestionPricingModel()
    voxel_id = "VOXEL-NYC-001"
    
    # Register Voxel: Low capacity implies high scarcity
    econ_engine.register_voxel(VoxelParams(
        voxel_id=voxel_id,
        base_price=10.0, # SkyTokens
        capacity=5
    ))
    
    print(f"[System] Registered {voxel_id} with Capacity=5")

    # 2. Setup Agents
    # Agent A: Medical Delivery (High Urgency, High Budget)
    agent_med = UAVAgent(agent_id="UAV-MED-99", budget=500.0, urgency=2.0)
    
    # Agent B: Cargo Logistics (Low Urgency, Strict Budget)
    agent_cargo = UAVAgent(agent_id="UAV-LOG-01", budget=100.0, urgency=0.8)
    
    agents = [agent_med, agent_cargo]
    print(f"[System] Agents Online: {[a.id for a in agents]}\n")

    # 3. Simulation Loop (Single Tick)
    current_occupancy = 4 # Almost full
    
    # Calculate current market price via Math Model
    current_toll = econ_engine.calculate_toll(voxel_id, current_occupancy)
    print(f"[Market] {voxel_id} Current Occupancy: {current_occupancy}/5")
    print(f"[Market] Dynamic Toll Price $P(v)$: {current_toll:.2f} SkyTokens\n")

    # 4. Agent Observation & Bidding
    bids = {}
    
    # Construct a synthetic observation vector
    # [occupancy, capacity, x, y, z, price]
    obs_vector = [current_occupancy, 5, 0, 0, 0, current_toll] 
    import numpy as np
    obs = np.array(obs_vector)

    for agent in agents:
        decision = agent.act(obs)
        bid = decision.get("bid", 0.0)
        action = decision.get("action")
        
        print(f"[{agent.id}] Urgency={agent.urgency} | Action={action} | Bid={bid:.2f}")
        
        if action == "MOVE" and bid > 0:
            bids[agent.id] = bid

    # 5. Auction Clearing (Vickrey-ish logic for demo)
    print("\n>>> [Auction] Clearing Bids...")
    if not bids:
        print("No bids placed.")
        return

    # Sort by bid price
    sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
    winner_id, winning_bid = sorted_bids[0]
    
    print(f"WINNER: {winner_id} with Bid {winning_bid:.2f}")
    
    # 6. Simulate Smart Contract Trigger
    print("\n>>> [L3 Protocol] Triggering Smart Contract...")
    print(f"Tx: Minting SkyRouteNFT(to={winner_id}, voxel={voxel_id}, price={winning_bid})")
    print("Tx: Status -> PENDING ORACLE VERIFICATION")
    print("Tx: Hash -> 0x7f9...3a1")

if __name__ == "__main__":
    asyncio.run(run_auction_simulation())

