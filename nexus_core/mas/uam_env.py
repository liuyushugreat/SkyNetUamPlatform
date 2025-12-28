import gym
import numpy as np
import uuid
from gym import spaces
from typing import List, Tuple, Dict, Optional
from nexus_core.assetization.event_bus import AssetEventBus, TelemetryEvent

class KinematicsEngine:
    """
    负责无人机物理计算的核心引擎。
    """
    def __init__(self, dt: float = 0.1, max_speed: float = 20.0, world_size: float = 500.0):
        self.dt = dt
        self.max_speed = max_speed
        self.world_size = world_size
        self.collision_radius = 5.0 

    def update(self, pos: np.ndarray, vel: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据动作更新状态。
        Action: 目标速度向量 (简化物理模型)
        """
        target_vel = action * self.max_speed
        # 惯性模拟
        new_vel = 0.8 * vel + 0.2 * target_vel
        new_vel = np.clip(new_vel, -self.max_speed, self.max_speed)
        
        new_pos = pos + new_vel * self.dt
        new_pos = np.clip(new_pos, 0, self.world_size)
        
        return new_pos, new_vel

    def check_collisions(self, positions: np.ndarray) -> List[int]:
        """
        检测 Agent 之间的碰撞。
        """
        crashed_agents = []
        num_agents = len(positions)
        dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        # 找到距离小于半径的索引
        rows, _ = np.where(dists < self.collision_radius)
        crashed_agents.extend(np.unique(rows).tolist())
        return crashed_agents

class SkyNetMultiAgentEnv(gym.Env):
    """
    支持 MARL (MADDPG) 的多无人机协同避障环境。
    """
    metadata = {'render.modes': ['human', 'console']}

    def __init__(self, 
                 num_agents: int = 5, 
                 neighbor_k: int = 3, 
                 render_mode: Optional[str] = None):
        super(SkyNetMultiAgentEnv, self).__init__()
        
        self.num_agents = num_agents
        self.neighbor_k = neighbor_k
        self.render_mode = render_mode
        self.dim_p = 3 
        
        # Assetization Event Bus
        self.event_bus = AssetEventBus.get_instance()
        self.mission_id = uuid.uuid4().hex

        self.physics = KinematicsEngine(dt=0.1, max_speed=15.0, world_size=500.0)

        # 动作空间: [vx, vy, vz]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # 观测空间: Pos(3) + Vel(3) + Goal_Rel(3) + Neighbors_Rel(3*K)
        # 更新 Observation Space 为归一化后的范围 [-1, 1]
        obs_dim = 3 + 3 + 3 + (3 * self.neighbor_k)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # 状态存储
        self.agents_pos = np.zeros((self.num_agents, 3))
        self.agents_vel = np.zeros((self.num_agents, 3))
        self.agents_goal = np.zeros((self.num_agents, 3))
        self.last_actions = np.zeros((self.num_agents, 3)) # 记录上一步动作，用于平滑性计算
        
        self.steps = 0
        self.max_steps = 200

        # 定义静态障碍物 (位置, 半径)
        # 放在环境中间阻挡路径
        center = self.physics.world_size / 2
        self.obstacles = [
            (np.array([center, center, center]), 40.0),
            (np.array([center - 100, center - 100, center]), 30.0),
            (np.array([center + 100, center + 100, center]), 30.0),
        ]

    def reset(self):
        self.steps = 0
        self.agents_pos = np.random.rand(self.num_agents, 3) * self.physics.world_size
        self.agents_vel = np.zeros((self.num_agents, 3))
        self.agents_goal = np.random.rand(self.num_agents, 3) * self.physics.world_size
        self.last_actions = np.zeros((self.num_agents, 3))
        return self._get_all_obs()

    def step(self, actions: List[np.ndarray]):
        self.steps += 1
        actions = np.array(actions)
        
        rewards = []
        dones = []
        infos = {}

        # 1. 物理更新
        for i in range(self.num_agents):
            self.agents_pos[i], self.agents_vel[i] = self.physics.update(
                self.agents_pos[i], 
                self.agents_vel[i], 
                actions[i]
            )
            
            # --- [ASSETIZATION] Publish Telemetry Event ---
            # Decoupled via EventBus. The AssetPipeline will listen and process.
            event = TelemetryEvent(
                event_id=uuid.uuid4().hex,
                event_type="TELEMETRY_UPDATE",
                source_id=f"uav_{i}",
                payload={
                    "pos": self.agents_pos[i].tolist(),
                    "vel": self.agents_vel[i].tolist()
                }
            )
            self.event_bus.publish(event)
            # ----------------------------------------------

        # 2. 碰撞检测
        crashed_indices = self.physics.check_collisions(self.agents_pos)
        
        # 额外：检测与静态障碍物的碰撞
        for i in range(self.num_agents):
            for obs_pos, obs_radius in self.obstacles:
                if np.linalg.norm(self.agents_pos[i] - obs_pos) < (self.physics.collision_radius + obs_radius):
                    if i not in crashed_indices:
                        crashed_indices.append(i)

        # 3. 计算奖励 (调用专门的奖励函数)
        for i in range(self.num_agents):
            is_crashed = i in crashed_indices
            
            # 判断是否到达目标
            dist_to_goal = np.linalg.norm(self.agents_pos[i] - self.agents_goal[i])
            is_reached = dist_to_goal < 5.0
            
            # 计算该 Agent 的 Step Reward
            rew = self._compute_reward(i, self.agents_pos[i], actions[i], is_reached, is_crashed)
            rewards.append(rew)
            
            # Done 逻辑
            dones.append(is_reached or is_crashed)

        # 4. 更新历史动作
        self.last_actions = actions.copy()

        # 全局 Done
        if self.steps >= self.max_steps:
            dones = [True] * self.num_agents

        # --- [ASSETIZATION] Trigger Mission Complete ---
        # If any agent finishes (or all finish), trigger asset creation
        for i, done in enumerate(dones):
            if done:
                self.event_bus.publish(TelemetryEvent(
                    event_id=self.mission_id,
                    event_type="MISSION_COMPLETE",
                    source_id=f"uav_{i}",
                    payload={"status": "COMPLETED", "final_step": self.steps}
                ))
        # -----------------------------------------------

        if self.render_mode == 'human':
            self.render()

        next_states = self._get_all_obs()
        return next_states, rewards, dones, infos

    def _compute_reward(self, agent_idx, current_pos, action, is_reached, is_crashed):
        """
        复合奖励函数设计 (Composite Reward Function)
        """
        reward = 0.0
        
        # --- 1. 目标引导奖励 (Goal Guidance) ---
        # 使用负距离作为稠密奖励信号，引导 Agent 靠近目标
        # 越近 dist 越小，reward 越大(越接近0)
        dist_to_goal = np.linalg.norm(current_pos - self.agents_goal[agent_idx])
        reward -= 0.1 * dist_to_goal 

        # --- 2. 碰撞惩罚 (Collision Penalty) ---
        if is_crashed:
            return -100.0  # 发生碰撞直接给予巨大惩罚并结束，无需计算其他细微奖励

        # --- 3. 势场引导奖励 (Potential Field / Safety) ---
        # 创新点：对进入探测范围的障碍物施加“排斥力”形式的负奖励
        
        # 3a. 针对其他无人机
        detection_range = 30.0
        for j in range(self.num_agents):
            if agent_idx == j: continue
            dist_j = np.linalg.norm(current_pos - self.agents_pos[j])
            if dist_j < detection_range:
                # 距离越近，惩罚呈指数或倒数增加
                # R_pot = - k * (1/d)
                # 为了防止除以零，加上一个小 epsilon
                penalty = 5.0 / (dist_j + 0.1)
                reward -= penalty

        # 3b. 针对静态障碍物
        for obs_pos, obs_radius in self.obstacles:
            dist_obs = np.linalg.norm(current_pos - obs_pos) - obs_radius
            if dist_obs < detection_range and dist_obs > 0:
                # 同样施加排斥惩罚
                penalty = 5.0 / (dist_obs + 0.1)
                reward -= penalty

        # --- 4. 能耗与平滑性 (Smoothness) ---
        # 惩罚动作幅度 (能耗)
        reward -= 0.05 * np.linalg.norm(action)
        
        # 惩罚加速度变化 (平滑性)
        # 鼓励轨迹平滑，不要剧烈晃动
        if self.last_actions is not None:
            delta_action = np.linalg.norm(action - self.last_actions[agent_idx])
            reward -= 0.1 * delta_action

        # --- 5. 任务完成奖励 (Completion) ---
        if is_reached:
            reward += 200.0
        
        # --- 6. 额外的时间步惩罚 ---
        # 鼓励尽快到达，每一步都扣分
        reward -= 0.1

        return reward

    def _get_all_obs(self):
        return [self._get_agent_obs(i) for i in range(self.num_agents)]

    def _get_agent_obs(self, agent_idx):
        # 归一化常量
        world_size = self.physics.world_size
        max_speed = self.physics.max_speed
        
        # 归一化自身状态
        pos = self.agents_pos[agent_idx] / world_size # [0, 1]
        vel = self.agents_vel[agent_idx] / max_speed # [-1, 1]
        
        # 归一化相对目标距离 (除以世界尺寸，使其落在 [-1, 1] 范围内)
        goal_rel = (self.agents_goal[agent_idx] - self.agents_pos[agent_idx]) / world_size
        
        # 感知逻辑: 找到最近的 K 个邻居
        other_pos = np.delete(self.agents_pos, agent_idx, axis=0)
        rel_positions = (other_pos - self.agents_pos[agent_idx]) / world_size # 归一化相对位置
        dists = np.linalg.norm(rel_positions, axis=1)
        
        # 排序
        sorted_indices = np.argsort(dists)
        nearest_k_indices = sorted_indices[:self.neighbor_k]
        
        # 填充邻居信息
        neighbor_obs = []
        for idx in nearest_k_indices:
            neighbor_obs.extend(rel_positions[idx])
            
        # 固定维度处理：如果邻居不足 K 个，用 0.0 填充
        # MADDPG 需要固定长度的输入向量
        padding = (3 * self.neighbor_k) - len(neighbor_obs)
        if padding > 0:
            neighbor_obs.extend([0.0] * padding)
            
        obs = np.concatenate([pos, vel, goal_rel, np.array(neighbor_obs)])
        
        # 确保数据类型正确并尽可能限制在 [-1, 1] 或 [0, 1]
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def render(self, mode='human'):
        print(f"Step {self.steps}")
