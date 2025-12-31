# Multi-Agent RL Framework for IJCAI 2026

This directory contains an upgraded experimental framework for large-scale multi-agent reinforcement learning, designed to meet IJCAI 2026 standards.

## 🎯 Key Features

### 1. **State-of-the-Art Algorithms**
- **MAPPO**: Multi-Agent Proximal Policy Optimization with attention-based critic
- **AP-MADDPG**: Existing implementation (from `nexus_core/mas/`)
- Both algorithms share the same attention-based critic architecture for fair comparison

### 2. **Emergence Metrics** (Critical for IJCAI)
- **Velocity Alignment (Order Parameter)**: Measures coordinated motion
- **Cluster Entropy**: Quantifies spatial grouping behavior
- Automatically logged to TensorBoard and CSV

### 3. **Massive Scalability**
- Vectorized environments supporting 50,000+ agents
- Parallel environment execution
- GPU-accelerated batched operations

## 📁 File Structure

```
research/experiments/maddpg/
├── networks.py              # Neural network architectures
│   ├── ActorNetwork         # Policy network (shared)
│   ├── AttentionCriticNetwork  # Attention-based critic (AP-MADDPG & MAPPO)
│   └── StandardCriticNetwork   # Baseline critic (for comparison)
│
├── mappo.py                 # MAPPO implementation
│   ├── MAPPOAgent          # Individual agent
│   └── MAPPOTrainer        # Centralized trainer with CTDE
│
├── emergence_metrics.py     # Emergence quantification
│   ├── calculate_velocity_alignment()  # Order parameter
│   ├── calculate_cluster_entropy()      # Spatial clustering
│   └── calculate_emergence_metrics()    # Combined metrics
│
├── metrics_logger.py        # Unified logging
│   └── MetricsLogger       # TensorBoard + CSV logging
│
├── vectorized_env.py        # Parallel environment execution
│   ├── VectorizedEnv       # Basic vectorization
│   └── BatchedVectorizedEnv # GPU-optimized batching
│
└── train.py                # Main training script
```

## 🚀 Quick Start

### Training MAPPO

```bash
# Basic training (1000 agents, 1 environment)
python train.py --algorithm MAPPO --num_agents 1000 --num_episodes 1000

# Large-scale training (50k agents, 10 parallel environments)
python train.py --algorithm MAPPO --num_agents 5000 --num_envs 10 --num_episodes 2000

# With attention-based critic
python train.py --algorithm MAPPO --num_agents 10000 --use_attention
```

### Training AP-MADDPG

```bash
# Use existing implementation in nexus_core/mas/
# Or integrate with this framework (see integration guide below)
```

## 📊 Emergence Metrics Explained

### Velocity Alignment (Order Parameter)
- **Range**: 0 (random motion) to 1 (perfect alignment)
- **Formula**: φ = (1/N) * |Σ_i v_i / |v_i||
- **Interpretation**: Higher values indicate coordinated swarm motion

### Cluster Entropy
- **Range**: 0 (few large clusters) to 1 (uniform cluster sizes)
- **Method**: DBSCAN clustering + entropy of cluster size distribution
- **Interpretation**: Measures spatial organization patterns

## 🔧 Architecture Details

### Attention-Based Critic (Shared by AP-MADDPG & MAPPO)

The attention mechanism enables "selective interaction" by:
1. Computing Query, Key, Value embeddings for each agent
2. Using multi-head attention to focus on critical neighbors
3. Reducing effective input dimension from N to K (key neighbors)

This architecture is crucial for scalability to 50k+ agents.

### MAPPO vs AP-MADDPG

| Feature | MAPPO | AP-MADDPG |
|---------|-------|-----------|
| Policy Update | PPO (clipped objective) | DDPG (deterministic) |
| Value Function | Shared attention-based critic | Per-agent attention-based critic |
| Exploration | Stochastic policy | Action noise |
| Sample Efficiency | On-policy (less efficient) | Off-policy (more efficient) |
| Stability | More stable | Requires careful tuning |

## 📈 Logging and Visualization

### TensorBoard
```bash
tensorboard --logdir=./logs/tensorboard
```

### CSV Metrics
Metrics are automatically saved to `logs/metrics_YYYYMMDD_HHMMSS.csv` with columns:
- Standard RL: `reward_mean`, `collision_rate`, `success_rate`, `actor_loss`, `critic_loss`
- Emergence: `order_parameter`, `local_alignment`, `cluster_entropy`, `num_clusters`

## 🔬 Integration with Existing Code

### Using AP-MADDPG with New Metrics

```python
from nexus_core.mas.maddpg import MADDPGTrainer, MADDPGAgent
from research.experiments.maddpg.emergence_metrics import calculate_emergence_metrics
from research.experiments.maddpg.metrics_logger import MetricsLogger

# Your existing MADDPG training loop
# ... collect states during evaluation ...

# Calculate emergence metrics
states = extract_states_from_obs(obs_list, num_agents)
metrics = calculate_emergence_metrics(states)

# Log metrics
logger.log_emergence_metrics(metrics, step=episode, algorithm='AP-MADDPG')
```

## 🎓 For IJCAI Submission

### Required Comparisons
1. **Baselines**: DDPG, MADDPG, MAPPO (this implementation)
2. **Your Method**: AP-MADDPG
3. **Metrics**: Reward, Collision Rate, Success Rate, **Emergence Metrics**

### Key Selling Points
1. **Scalability**: 50k-100k agents (demonstrated via vectorized envs)
2. **Emergence**: Quantified via velocity alignment and cluster entropy
3. **Fair Comparison**: All methods use same attention-based critic architecture

## 🐛 Troubleshooting

### Out of Memory
- Reduce `num_envs` or `num_agents_per_env`
- Use CPU instead of GPU: set `device='cpu'` in `BatchedVectorizedEnv`

### Slow Training
- Increase `num_envs` for better parallelization
- Use `BatchedVectorizedEnv` for GPU acceleration
- Reduce `UPDATE_EPOCHS` in `mappo.py` if needed

## 📚 References

1. **MAPPO**: Yu, C., et al. (2021). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. NeurIPS.

2. **MADDPG**: Lowe, R., et al. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. NIPS.

3. **Emergence Metrics**: Vicsek, T., et al. (1995). Novel type of phase transition in a system of self-driven particles. Physical Review Letters.

## 📝 License

Same as parent project (Apache 2.0).

