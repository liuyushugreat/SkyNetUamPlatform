# Quick Start Guide

## 🎯 Goal
Upgrade your experimental framework to meet IJCAI 2026 standards with:
1. MAPPO baseline (state-of-the-art)
2. Emergence metrics (crucial for acceptance)
3. Scalability to 50k+ agents

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies
```bash
pip install torch torchvision scikit-learn scipy
pip install tensorboard  # Optional but recommended
```

### Step 2: Test MAPPO on Small Scale
```bash
cd research/experiments/maddpg
python train.py --algorithm MAPPO --num_agents 100 --num_episodes 100
```

### Step 3: View Results
```bash
# TensorBoard (if installed)
tensorboard --logdir=./logs/tensorboard

# Or check CSV
cat logs/metrics_*.csv | tail -20
```

## 📊 Understanding the Output

### Training Output
```
Episode 10/1000: Reward=45.23, Collision Rate=0.0234, 
                 Order Param=0.456, Cluster Entropy=0.623
```

**What This Means:**
- **Reward**: Higher is better (agent performance)
- **Collision Rate**: Lower is better (safety)
- **Order Parameter**: 0-1, higher = more coordinated motion (emergence!)
- **Cluster Entropy**: 0-1, measures spatial organization

### Logged Files
```
logs/
  ├── tensorboard/          # TensorBoard logs
  └── metrics_YYYYMMDD_HHMMSS.csv  # All metrics in CSV
```

## 🔬 Adding Emergence Metrics to Your Existing Code

### Option 1: Minimal Integration (5 lines)

```python
from research.experiments.maddpg.emergence_metrics import calculate_emergence_metrics

# In your evaluation loop, after collecting states:
states = np.array([[x, y, z, vx, vy, vz, ...] for agent in agents])
metrics = calculate_emergence_metrics(states)

print(f"Emergence: Order={metrics['order_parameter']:.3f}, "
      f"Entropy={metrics['cluster_entropy']:.3f}")
```

### Option 2: Full Integration with Logging

```python
from research.experiments.maddpg.emergence_metrics import calculate_emergence_metrics
from research.experiments.maddpg.metrics_logger import MetricsLogger

logger = MetricsLogger(log_dir='./logs')

# In your training loop:
for episode in range(num_episodes):
    # ... your training code ...
    
    # Calculate emergence metrics
    states = extract_states(observations)
    emergence = calculate_emergence_metrics(states)
    
    # Log everything
    logger.log_episode({
        'reward_mean': avg_reward,
        'collision_rate': collision_rate,
        **emergence  # Include all emergence metrics
    }, algorithm='AP-MADDPG')
    
    logger.log_emergence_metrics(emergence, episode, algorithm='AP-MADDPG')
```

## 🚀 Scaling to 50k Agents

### Strategy 1: Parallel Environments (Recommended)
```python
# 10 environments × 5000 agents = 50,000 total
python train.py --algorithm MAPPO --num_agents 5000 --num_envs 10
```

### Strategy 2: Single Large Environment
```python
# 1 environment × 50000 agents (if env supports)
python train.py --algorithm MAPPO --num_agents 50000 --num_envs 1
```

### Strategy 3: Gradual Scale-Up
```python
# Train progressively larger swarms
for scale in [1000, 5000, 10000, 25000, 50000]:
    python train.py --algorithm MAPPO --num_agents $scale --num_episodes 500
```

## 📈 Generating Paper Figures

### From CSV Logs
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv('logs/metrics_20250101_120000.csv')

# Plot order parameter over episodes
plt.figure()
plt.plot(df['episode'], df['order_parameter'])
plt.xlabel('Episode')
plt.ylabel('Order Parameter')
plt.title('Emergence of Coordinated Motion')
plt.savefig('order_parameter.png')

# Plot collision rate vs swarm size
# (aggregate data from multiple runs)
```

### From TensorBoard
```bash
# Export data
tensorboard --logdir=./logs/tensorboard --export_to_png=./figures
```

## 🔍 Comparing Algorithms

### Run All Baselines
```bash
# MAPPO
python train.py --algorithm MAPPO --num_agents 10000 --num_episodes 2000

# AP-MADDPG (use existing code, add metrics logging)
# See integration examples above

# DDPG, MADDPG (use existing implementations)
```

### Compare Results
```python
import pandas as pd

# Load all algorithm logs
mappo_df = pd.read_csv('logs/mappo_metrics.csv')
maddpg_df = pd.read_csv('logs/maddpg_metrics.csv')
ap_maddpg_df = pd.read_csv('logs/ap_maddpg_metrics.csv')

# Compare order parameters
print("Order Parameter Comparison:")
print(f"MAPPO: {mappo_df['order_parameter'].mean():.3f}")
print(f"MADDPG: {maddpg_df['order_parameter'].mean():.3f}")
print(f"AP-MADDPG: {ap_maddpg_df['order_parameter'].mean():.3f}")
```

## 🐛 Common Issues

### Issue: "Module not found"
**Solution**: Add parent directory to Python path
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))  # repo root
```

### Issue: "Out of memory"
**Solution**: Reduce batch size or use CPU
```python
# In train.py, reduce BATCH_SIZE or set device='cpu'
```

### Issue: "Slow training"
**Solution**: Use more parallel environments
```bash
python train.py --num_envs 10  # Instead of 1
```

## ✅ Verification Checklist

After setup, verify:

- [ ] MAPPO trains successfully on 100 agents
- [ ] Emergence metrics are calculated (non-zero values)
- [ ] TensorBoard shows metrics (or CSV file created)
- [ ] Can scale to 1000+ agents
- [ ] Integration with existing code works

## 📚 Next Steps

1. **Read**: `IMPLEMENTATION_SUMMARY.md` for architecture details
2. **Run**: Small-scale experiments to verify setup
3. **Scale**: Gradually increase to 50k agents
4. **Compare**: Run all algorithms with same protocol
5. **Analyze**: Generate figures from logged metrics
6. **Write**: Include emergence metrics in paper

## 💡 Pro Tips

1. **Start Small**: Always test on 100 agents before scaling
2. **Monitor Emergence**: Order parameter should increase over training
3. **Log Everything**: You'll need metrics for paper figures
4. **Fair Comparison**: Use same attention architecture for all algorithms
5. **Statistical Tests**: Run 10+ seeds for significance tests

## 🆘 Need Help?

- Check `IMPLEMENTATION_SUMMARY.md` for architecture details
- Check `README.md` for full documentation
- Review code comments in each module
- Check TensorBoard logs for training progress

Good luck with your IJCAI submission! 🎓

