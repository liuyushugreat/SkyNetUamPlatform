"""
Training Script for Multi-Agent RL Algorithms.

Supports:
- MAPPO (with attention-based critic)
- AP-MADDPG (existing implementation)
- Vectorized environments for 50k+ agents
- Emergence metrics logging

Usage:
    python train.py --algorithm MAPPO --num_agents 50000 --num_envs 10
"""

import argparse
import numpy as np
import torch
import os
from typing import Dict, List
import sys

# Add repo root directory to sys.path for imports.
# This script lives at: <repo_root>/PaperExperiments/experimentsMADDPG/train.py
# We need repo_root on sys.path so that `import PaperExperiments.experimentsMADDPG.*` works.
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_root)

from PaperExperiments.experimentsMADDPG.mappo import MAPPOAgent, MAPPOTrainer
from PaperExperiments.experimentsMADDPG.emergence_metrics import calculate_emergence_metrics
from PaperExperiments.experimentsMADDPG.metrics_logger import MetricsLogger
from PaperExperiments.experimentsMADDPG.vectorized_env import VectorizedEnv, BatchedVectorizedEnv
from nexus_core.mas.environment import SkyNetEnv


def create_env(num_agents: int = 100):
    """Factory function to create environment instances."""
    return SkyNetEnv(num_agents=num_agents)


def extract_states_from_obs(obs_list: List[Dict], num_agents: int) -> np.ndarray:
    """
    Extract state array from observation dictionaries.
    
    Assumes observation contains [x, y, z, vx, vy, vz, ...]
    """
    states = []
    for obs_dict in obs_list:
        agent_keys = sorted(obs_dict.keys())
        for key in agent_keys[:num_agents]:
            obs = obs_dict[key]
            # Extract position and velocity (first 6 elements)
            state = obs[:6] if len(obs) >= 6 else np.pad(obs, (0, 6 - len(obs)))
            states.append(state)
    
    return np.array(states)


def train_mappo(num_agents: int = 1000, num_envs: int = 1,
                num_episodes: int = 1000, max_steps: int = 500,
                log_dir: str = './logs', use_attention: bool = True):
    """
    Train MAPPO algorithm.
    
    Args:
        num_agents: Number of agents per environment
        num_envs: Number of parallel environments
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        log_dir: Directory for logging
        use_attention: Use attention-based critic
    """
    print(f"Training MAPPO with {num_agents} agents per env, {num_envs} parallel envs")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create vectorized environment
    vectorized_env = VectorizedEnv(
        env_factory=lambda: create_env(num_agents),
        num_envs=num_envs,
        num_agents_per_env=num_agents
    )
    
    # Get environment dimensions
    sample_obs = vectorized_env.reset()[0]
    sample_agent_key = list(sample_obs.keys())[0]
    obs_dim = sample_obs[sample_agent_key].shape[0]
    action_dim = vectorized_env.envs[0].action_spaces[sample_agent_key].shape[0]
    
    # Create agents
    agents = []
    for i in range(num_agents):
        agent = MAPPOAgent(
            agent_id=f"uav_{i}",
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            use_attention=use_attention
        )
        agents.append(agent)
    
    # Create trainer
    trainer = MAPPOTrainer(
        agents=agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        use_attention=use_attention
    )
    
    # Create logger
    logger = MetricsLogger(log_dir=log_dir, use_tensorboard=True)
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    collision_counts = []
    
    for episode in range(num_episodes):
        # Reset environments
        obs_list = vectorized_env.reset()
        global_states = vectorized_env.get_global_states()
        
        episode_reward = np.zeros(num_agents)
        episode_length = 0
        collisions = 0
        
        for step in range(max_steps):
            # Collect actions from all agents
            actions_list = []
            log_probs_list = []
            values_list = []
            
            for env_idx, obs_dict in enumerate(obs_list):
                actions_dict = {}
                log_probs_dict = {}
                values_dict = {}
                
                # Get actions from agents
                agent_keys = sorted(obs_dict.keys())
                for agent_idx, agent_key in enumerate(agent_keys):
                    obs = torch.FloatTensor(obs_dict[agent_key]).to(device)
                    agent = agents[agent_idx]
                    
                    action, log_prob = agent.get_action(obs)
                    actions_dict[agent_key] = action.cpu().numpy()
                    log_probs_dict[agent_key] = log_prob.cpu().item()
                    
                    # Compute value estimate
                    if trainer.use_attention:
                        obs_n = [torch.FloatTensor(obs_dict[k]).to(device) for k in agent_keys]
                        action_n = [torch.FloatTensor(actions_dict[k]).to(device) for k in agent_keys]
                        value, _ = trainer.critic(obs_n, action_n, agent_idx)
                    else:
                        global_state = torch.FloatTensor(global_states[env_idx]).to(device)
                        actions_cat = torch.cat([torch.FloatTensor(actions_dict[k]).to(device) 
                                                for k in agent_keys])
                        value = trainer.critic(global_state.unsqueeze(0), actions_cat.unsqueeze(0))
                    
                    values_dict[agent_key] = value.cpu().item()
                
                actions_list.append(actions_dict)
                log_probs_list.append(log_probs_dict)
                values_list.append(values_dict)
            
            # Step environments
            next_obs_list, rewards_list, dones_list, infos_list = vectorized_env.step(actions_list)
            next_global_states = vectorized_env.get_global_states()
            
            # Store transitions
            for env_idx in range(num_envs):
                obs_n = [torch.FloatTensor(obs_list[env_idx][k]) for k in sorted(obs_list[env_idx].keys())]
                action_n = [torch.FloatTensor(actions_list[env_idx][k]) for k in sorted(actions_list[env_idx].keys())]
                reward_n = [rewards_list[env_idx][k] for k in sorted(rewards_list[env_idx].keys())]
                log_prob_n = [log_probs_list[env_idx][k] for k in sorted(log_probs_list[env_idx].keys())]
                value_n = [values_list[env_idx][k] for k in sorted(values_list[env_idx].keys())]
                done_n = [dones_list[env_idx].get(k, False) for k in sorted(obs_list[env_idx].keys())]
                global_state = global_states[env_idx]
                
                trainer.store_transition(
                    obs_n, action_n, reward_n, log_prob_n, value_n, done_n, global_state
                )
                
                # Accumulate rewards
                for agent_idx, key in enumerate(sorted(obs_list[env_idx].keys())):
                    episode_reward[agent_idx] += reward_n[agent_idx]
                    if 'collision' in infos_list[env_idx].get(key, {}):
                        collisions += 1
            
            episode_length += 1
            obs_list = next_obs_list
            global_states = next_global_states
            
            # Check if all environments are done
            all_done = all(dones_list[i].get('__all__', False) for i in range(num_envs))
            if all_done:
                break
        
        # Update policy
        update_metrics = trainer.update()
        
        # Calculate emergence metrics (on first environment)
        states = extract_states_from_obs([obs_list[0]], num_agents)
        emergence_metrics = calculate_emergence_metrics(states)
        
        # Log metrics
        avg_reward = np.mean(episode_reward)
        collision_rate = collisions / (num_agents * episode_length) if episode_length > 0 else 0.0
        
        metrics = {
            'reward_mean': avg_reward,
            'collision_rate': collision_rate,
            'episode_length': episode_length,
            **update_metrics,
            **emergence_metrics
        }
        
        logger.log_episode(metrics, algorithm='MAPPO')
        logger.log_emergence_metrics(emergence_metrics, episode, algorithm='MAPPO')
        
        episode_rewards.append(avg_reward)
        episode_lengths.append(episode_length)
        collision_counts.append(collisions)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Reward={avg_reward:.2f}, "
                  f"Collision Rate={collision_rate:.4f}, "
                  f"Order Param={emergence_metrics['order_parameter']:.3f}, "
                  f"Cluster Entropy={emergence_metrics['cluster_entropy']:.3f}")
    
    logger.close()
    print(f"Training complete. Logs saved to {log_dir}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'collision_counts': collision_counts
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Multi-Agent RL Algorithms')
    parser.add_argument('--algorithm', type=str, default='MAPPO',
                       choices=['MAPPO', 'AP-MADDPG'],
                       help='Algorithm to train')
    parser.add_argument('--num_agents', type=int, default=1000,
                       help='Number of agents per environment')
    parser.add_argument('--num_envs', type=int, default=1,
                       help='Number of parallel environments')
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for logging')
    parser.add_argument('--use_attention', action='store_true',
                       help='Use attention-based critic')
    
    args = parser.parse_args()
    
    if args.algorithm == 'MAPPO':
        train_mappo(
            num_agents=args.num_agents,
            num_envs=args.num_envs,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            log_dir=args.log_dir,
            use_attention=args.use_attention
        )
    else:
        print(f"Algorithm {args.algorithm} not yet implemented in this script.")
        print("Please use the existing implementation in nexus_core/mas/")

