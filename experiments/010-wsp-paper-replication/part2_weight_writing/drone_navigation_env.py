import os
import json
import numpy as np

class DroneObstacleEnv:
    """
    Continuous 2D Drone Obstacle Avoidance & Goal Seeking Environment.
    State [4]:
      0: d_left  (distance to left obstacle, 0..10)
      1: d_right (distance to right obstacle, 0..10)
      2: x_goal  (lateral distance to goal, -5..+5)
      3: vx      (lateral velocity, -3..+3)
    Action [1]:
      ax         (steering acceleration, clipped to [-2.0, +2.0])
    """
    def __init__(self, dt=0.1, max_steps=80):
        self.dt = dt
        self.max_steps = max_steps
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.x = 0.0
        self.vx = np.random.uniform(-0.5, 0.5)
        # Goal lateral position
        self.x_goal = np.random.uniform(-4.0, 4.0)
        # Obstacles: left wall at x_left, right wall at x_right
        self.x_left = -np.random.uniform(1.5, 3.5)
        self.x_right = np.random.uniform(1.5, 3.5)
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        d_left = max(0.0, self.x - self.x_left)
        d_right = max(0.0, self.x_right - self.x)
        rel_goal = self.x_goal - self.x
        return np.array([d_left, d_right, rel_goal, self.vx], dtype=np.float32)

    def step(self, action):
        ax = np.clip(float(action), -2.0, 2.0)
        self.vx += ax * self.dt
        self.vx = np.clip(self.vx, -3.0, 3.0)
        self.x += self.vx * self.dt
        self.step_count += 1

        obs = self._get_obs()
        d_left, d_right, rel_goal, _ = obs

        # Check collision
        collision = (d_left <= 0.05) or (d_right <= 0.05)
        goal_reached = abs(rel_goal) <= 0.3
        done = collision or goal_reached or (self.step_count >= self.max_steps)

        # Reward formulation:
        # Distance to goal penalty + large collision penalty + goal bonus
        reward = -abs(rel_goal) * 0.1
        if collision:
            reward -= 50.0
        if goal_reached:
            reward += 100.0

        return obs, reward, done, {
            "collision": collision,
            "goal_reached": goal_reached,
            "steps": self.step_count,
            "final_dist": abs(rel_goal)
        }

def evaluate_network(W1, b1, W2, b2, n_episodes=50, activation="relu"):
    """
    Evaluates policy MLP[4 -> 8 -> 1] across n_episodes.
    W1: [8, 4], b1: [8]
    W2: [1, 8], b2: [1]
    """
    env = DroneObstacleEnv()
    total_rewards = []
    collisions = 0
    goals_reached = 0
    final_distances = []

    for ep in range(n_episodes):
        obs = env.reset(seed=1000 + ep)
        done = False
        ep_reward = 0.0

        while not done:
            # Forward pass
            h = np.dot(W1, obs) + b1
            if activation == "relu":
                h_act = np.maximum(0.0, h)
            elif activation == "tanh":
                h_act = np.tanh(h)
            else:
                h_act = h
            action = float((np.dot(W2, h_act) + b2).item())

            obs, r, done, info = env.step(action)
            ep_reward += r

        total_rewards.append(ep_reward)
        if info["collision"]:
            collisions += 1
        if info["goal_reached"]:
            goals_reached += 1
        final_distances.append(info["final_dist"])

    return {
        "n_episodes": n_episodes,
        "mean_reward": round(float(np.mean(total_rewards)), 2),
        "collision_rate_pct": round(float(collisions / n_episodes) * 100, 1),
        "goal_reached_rate_pct": round(float(goals_reached / n_episodes) * 100, 1),
        "mean_final_dist": round(float(np.mean(final_distances)), 3),
        "survival_rate_pct": round(float((n_episodes - collisions) / n_episodes) * 100, 1)
    }

if __name__ == "__main__":
    # Smoke test with random weights
    W1 = np.random.randn(8, 4) * 0.1
    b1 = np.zeros(8)
    W2 = np.random.randn(1, 8) * 0.1
    b2 = np.zeros(1)
    res = evaluate_network(W1, b1, W2, b2, n_episodes=10)
    print("Random Baseline Test:", res)
