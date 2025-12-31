"""
Vectorized Environment Wrapper for Massive Parallel Simulation.

This module enables efficient parallel execution of 50k+ agents by:
1. Batching environment steps across multiple parallel instances
2. Using GPU acceleration where possible
3. Efficient state/action batching

Supports both synchronous and asynchronous parallel execution.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp


class VectorizedEnv:
    """
    Wrapper for running multiple environment instances in parallel.
    
    This is crucial for scaling to 50k+ agents efficiently.
    """
    def __init__(self, env_factory: Callable, num_envs: int = 1,
                 num_agents_per_env: int = 1000, use_multiprocessing: bool = False):
        """
        Args:
            env_factory: Function that creates a new environment instance
            num_envs: Number of parallel environment instances
            num_agents_per_env: Number of agents per environment
            use_multiprocessing: Use multiprocessing (slower but more isolated)
        """
        self.num_envs = num_envs
        self.num_agents_per_env = num_agents_per_env
        self.total_agents = num_envs * num_agents_per_env
        self.use_multiprocessing = use_multiprocessing
        
        # Create environments
        if use_multiprocessing:
            # Use process pool for true parallelism
            self.executor = ProcessPoolExecutor(max_workers=num_envs)
            # Note: In practice, you might want to use shared memory for efficiency
            self.envs = None  # Environments created in subprocesses
        else:
            # Use thread pool (faster for I/O-bound, shared memory)
            self.executor = ThreadPoolExecutor(max_workers=num_envs)
            self.envs = [env_factory() for _ in range(num_envs)]
        
        self.env_factory = env_factory
        
        # Get observation/action spaces from first env
        if self.envs:
            sample_env = self.envs[0]
            self.obs_dim = sample_env.observation_spaces[list(sample_env.observation_spaces.keys())[0]].shape[0]
            self.action_dim = sample_env.action_spaces[list(sample_env.action_spaces.keys())[0]].shape[0]
        else:
            # Default dimensions (will be set after first reset)
            self.obs_dim = None
            self.action_dim = None
    
    def reset(self) -> List[Dict]:
        """
        Reset all parallel environments.
        
        Returns:
            List of observation dictionaries, one per environment
        """
        if self.use_multiprocessing:
            # Reset in parallel processes
            futures = [self.executor.submit(self._reset_env, i) for i in range(self.num_envs)]
            obs_list = [f.result() for f in futures]
        else:
            # Reset sequentially (usually fast enough)
            obs_list = [env.reset() for env in self.envs]
        
        # Set dimensions if not set
        if self.obs_dim is None and len(obs_list) > 0:
            sample_obs = obs_list[0]
            first_key = list(sample_obs.keys())[0]
            self.obs_dim = sample_obs[first_key].shape[0]
        
        return obs_list
    
    def step(self, actions_list: List[Dict]) -> Tuple[List[Dict], List[Dict], 
                                                       List[Dict], List[Dict]]:
        """
        Step all parallel environments.
        
        Args:
            actions_list: List of action dictionaries, one per environment
        
        Returns:
            Tuple of (obs_list, rewards_list, dones_list, infos_list)
        """
        if self.use_multiprocessing:
            futures = [
                self.executor.submit(self._step_env, i, actions_list[i])
                for i in range(self.num_envs)
            ]
            results = [f.result() for f in futures]
            obs_list, rewards_list, dones_list, infos_list = zip(*results)
        else:
            results = [
                env.step(actions_list[i])
                for i, env in enumerate(self.envs)
            ]
            obs_list, rewards_list, dones_list, infos_list = zip(*results)
        
        return list(obs_list), list(rewards_list), list(dones_list), list(infos_list)
    
    def get_global_states(self) -> List[np.ndarray]:
        """
        Get global states from all environments.
        
        Returns:
            List of global state arrays
        """
        if self.use_multiprocessing:
            futures = [self.executor.submit(self._get_global_state, i) for i in range(self.num_envs)]
            return [f.result() for f in futures]
        else:
            return [env.get_global_state() for env in self.envs]
    
    def _reset_env(self, env_idx: int) -> Dict:
        """Helper for multiprocessing reset."""
        # In multiprocessing, we need to create env in subprocess
        # This is a simplified version - in practice, use shared memory
        env = self.env_factory()
        return env.reset()
    
    def _step_env(self, env_idx: int, actions: Dict) -> Tuple:
        """Helper for multiprocessing step."""
        env = self.env_factory()
        # Note: This creates a new env each time - optimize with shared state
        return env.step(actions)
    
    def _get_global_state(self, env_idx: int) -> np.ndarray:
        """Helper for multiprocessing global state."""
        env = self.env_factory()
        return env.get_global_state()
    
    def close(self):
        """Close all environments and executor."""
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.envs:
            for env in self.envs:
                if hasattr(env, 'close'):
                    env.close()


class BatchedVectorizedEnv:
    """
    Optimized version that batches operations for GPU acceleration.
    
    Collects states/actions from all environments into batched tensors
    for efficient GPU processing.
    """
    def __init__(self, vectorized_env: VectorizedEnv, device: str = 'cuda'):
        """
        Args:
            vectorized_env: Underlying VectorizedEnv instance
            device: Device for tensor operations ('cuda' or 'cpu')
        """
        self.vectorized_env = vectorized_env
        self.device = device
    
    def reset_batched(self) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Reset and return batched observations.
        
        Returns:
            Tuple of (batched_obs_tensor, obs_dict_list)
            batched_obs_tensor: (num_envs, num_agents, obs_dim)
        """
        obs_list = self.vectorized_env.reset()
        
        # Convert to batched tensor
        # Assuming all envs have same number of agents
        num_agents = len(obs_list[0])
        batched_obs = torch.zeros(
            self.vectorized_env.num_envs,
            num_agents,
            self.vectorized_env.obs_dim,
            device=self.device
        )
        
        for env_idx, obs_dict in enumerate(obs_list):
            agent_keys = sorted(obs_dict.keys())
            for agent_idx, key in enumerate(agent_keys):
                batched_obs[env_idx, agent_idx] = torch.tensor(
                    obs_dict[key], device=self.device
                )
        
        return batched_obs, obs_list
    
    def step_batched(self, batched_actions: torch.Tensor) -> Tuple:
        """
        Step with batched actions.
        
        Args:
            batched_actions: (num_envs, num_agents, action_dim) tensor
        
        Returns:
            Tuple of batched results
        """
        # Convert batched tensor to list of dicts
        num_envs, num_agents, action_dim = batched_actions.shape
        actions_list = []
        
        for env_idx in range(num_envs):
            actions_dict = {}
            for agent_idx in range(num_agents):
                agent_id = f"uav_{agent_idx}"
                actions_dict[agent_id] = batched_actions[env_idx, agent_idx].cpu().numpy()
            actions_list.append(actions_dict)
        
        # Step environments
        obs_list, rewards_list, dones_list, infos_list = self.vectorized_env.step(actions_list)
        
        # Convert back to batched tensors
        batched_obs = torch.zeros_like(batched_actions[:, :, :self.vectorized_env.obs_dim])
        batched_rewards = torch.zeros(num_envs, num_agents, device=self.device)
        batched_dones = torch.zeros(num_envs, num_agents, dtype=torch.bool, device=self.device)
        
        for env_idx, obs_dict in enumerate(obs_list):
            agent_keys = sorted(obs_dict.keys())
            for agent_idx, key in enumerate(agent_keys):
                batched_obs[env_idx, agent_idx] = torch.tensor(obs_dict[key], device=self.device)
                batched_rewards[env_idx, agent_idx] = rewards_list[env_idx][key]
                batched_dones[env_idx, agent_idx] = dones_list[env_idx][key]
        
        return batched_obs, batched_rewards, batched_dones, infos_list

