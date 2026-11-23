import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set fixed seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

OUT_DIR = "C:/Users/luisv/Projects/weightliftai/experiments/010-wsp-paper-replication/part1_weight_reading"
os.makedirs(OUT_DIR, exist_ok=True)

class StandardMLP(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=16, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

def train_parity():
    # 4-bit parity
    model = StandardMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()
    
    # 16 possible 4-bit combinations
    inputs = []
    labels = []
    for i in range(16):
        bits = [(i >> b) & 1 for b in range(4)]
        x = [1.0 if b == 1 else -1.0 for b in bits]
        parity = sum(bits) % 2
        inputs.append(x)
        labels.append(parity)
    
    X = torch.tensor(inputs, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.long)
    
    for epoch in range(400):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()
        
    acc = (torch.argmax(model(X), dim=1) == Y).float().mean().item()
    print(f"Task 0 (Parity) trained: loss={loss.item():.4f}, acc={acc*100:.1f}%")
    return model

def train_goal_seeking():
    # 2D Goal Seeking: steer towards (0,0)
    model = StandardMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Random 2D coordinates [x, y, vx, vy]
    X = torch.randn(500, 4)
    # Target force: proportional to -position - 0.5*velocity
    target_fx = -1.5 * X[:, 0] - 0.5 * X[:, 2]
    target_fy = -1.5 * X[:, 1] - 0.5 * X[:, 3]
    Y = torch.stack([target_fx, target_fy], dim=1)
    
    for epoch in range(300):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()
        
    print(f"Task 1 (Goal Seeking) trained: loss={loss.item():.4f}")
    return model

def train_pendulum_balance():
    # Inverted Pendulum: input [theta, theta_dot, x, x_dot]
    model = StandardMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Linear Quadratic Regulator (LQR) target approximation
    X = torch.randn(500, 4) * 0.5
    # Balances around theta=0: strong restoring torque on theta, damping on theta_dot
    u_control = -8.0 * X[:, 0] - 2.0 * X[:, 1] - 1.0 * X[:, 2] - 0.5 * X[:, 3]
    u_aux = 0.5 * X[:, 0] # auxiliary diagnostic monitor
    Y = torch.stack([u_control, u_aux], dim=1)
    
    for epoch in range(300):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()
        
    print(f"Task 2 (Pendulum Balance) trained: loss={loss.item():.4f}")
    return model

def train_frequency_discriminator():
    # 4-point sampled sine wave: classify low vs high frequency
    model = StandardMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    inputs = []
    labels = []
    t = np.linspace(0, 1, 4)
    for _ in range(500):
        is_high = np.random.choice([0, 1])
        freq = np.random.uniform(3.0, 5.0) if is_high else np.random.uniform(0.5, 1.5)
        phase = np.random.uniform(0, 2*np.pi)
        signal = np.sin(2 * np.pi * freq * t + phase) + np.random.randn(4) * 0.05
        inputs.append(signal)
        labels.append(is_high)
        
    X = torch.tensor(np.array(inputs), dtype=torch.float32)
    Y = torch.tensor(np.array(labels), dtype=torch.long)
    
    for epoch in range(300):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()
        
    acc = (torch.argmax(model(X), dim=1) == Y).float().mean().item()
    print(f"Task 3 (Frequency Discriminator) trained: loss={loss.item():.4f}, acc={acc*100:.1f}%")
    return model

def serialize_model_raw(model):
    """Strip all metadata and extract raw numpy arrays + spectral properties."""
    w1 = model.fc1.weight.detach().cpu().numpy() # [16, 4]
    b1 = model.fc1.bias.detach().cpu().numpy()   # [16]
    w2 = model.fc2.weight.detach().cpu().numpy() # [2, 16]
    b2 = model.fc2.bias.detach().cpu().numpy()   # [2]
    
    # Compute SVD
    _, s1, _ = np.linalg.svd(w1)
    _, s2, _ = np.linalg.svd(w2)
    
    return {
        "fc1_weight": np.round(w1, 3).tolist(),
        "fc1_bias": np.round(b1, 3).tolist(),
        "fc2_weight": np.round(w2, 3).tolist(),
        "fc2_bias": np.round(b2, 3).tolist(),
        "spectral_summary": {
            "fc1_singular_values": np.round(s1, 3).tolist(),
            "fc1_frobenius_norm": round(float(np.linalg.norm(w1)), 3),
            "fc2_singular_values": np.round(s2, 3).tolist(),
            "fc2_frobenius_norm": round(float(np.linalg.norm(w2)), 3),
            "fc1_row_norms": np.round(np.linalg.norm(w1, axis=1), 3).tolist(),
            "fc1_col_norms": np.round(np.linalg.norm(w1, axis=0), 3).tolist(),
        }
    }

def main():
    print("=== PART 1: TRAINING BENCHMARK NETWORKS FOR WEIGHT READING ===")
    m_parity = train_parity()
    m_goal = train_goal_seeking()
    m_pendulum = train_pendulum_balance()
    m_freq = train_frequency_discriminator()
    
    tasks = [
        {"id": "NET-A", "true_task": "Non-linear Boolean Parity Function", "true_key": "parity", "raw": serialize_model_raw(m_parity)},
        {"id": "NET-B", "true_task": "2D Spatial Coordinate Navigation / Goal Seeking", "true_key": "goal_seeking", "raw": serialize_model_raw(m_goal)},
        {"id": "NET-C", "true_task": "Inverted Pendulum Dynamical Balance (LQR Control)", "true_key": "pendulum_balance", "raw": serialize_model_raw(m_pendulum)},
        {"id": "NET-D", "true_task": "4-Point Sinusoidal Frequency Discriminator", "true_key": "frequency_discriminator", "raw": serialize_model_raw(m_freq)},
    ]
    
    # Shuffle into blind test set
    perm = np.random.permutation(4)
    blind_dataset = []
    gold_answers = {}
    
    TASK_OPTIONS = [
        "Option 1: Non-linear Boolean 4-Bit Parity Function",
        "Option 2: 2D Spatial Coordinate Navigation / Goal Seeking to Origin",
        "Option 3: Inverted Pendulum Dynamical Angle & Velocity Stabilizer (LQR)",
        "Option 4: 4-Point Sinusoidal Wave Frequency Discriminator"
    ]
    
    mapping = {
        "parity": "Option 1: Non-linear Boolean 4-Bit Parity Function",
        "goal_seeking": "Option 2: 2D Spatial Coordinate Navigation / Goal Seeking to Origin",
        "pendulum_balance": "Option 3: Inverted Pendulum Dynamical Angle & Velocity Stabilizer (LQR)",
        "frequency_discriminator": "Option 4: 4-Point Sinusoidal Wave Frequency Discriminator"
    }
    
    for idx, p_idx in enumerate(perm):
        item = tasks[p_idx]
        test_id = f"ANONYMOUS_NET_{idx+1}"
        blind_dataset.append({
            "anonymous_id": test_id,
            "architecture": "MLP[4 -> 16 -> 2] with Tanh activation",
            "raw_weights": item["raw"],
            "options": TASK_OPTIONS
        })
        gold_answers[test_id] = {
            "true_task": item["true_task"],
            "correct_option": mapping[item["true_key"]],
            "true_key": item["true_key"]
        }
        
    blind_file = os.path.join(OUT_DIR, "blind_weight_reading_test.json")
    with open(blind_file, "w", encoding="utf-8") as f:
        json.dump(blind_dataset, f, indent=2)
        
    gold_file = os.path.join(OUT_DIR, "gold_answers_sealed.json")
    with open(gold_file, "w", encoding="utf-8") as f:
        json.dump(gold_answers, f, indent=2)
        
    print(f"\nGenerated 4 blind test instances in {blind_file}")
    print(f"Sealed gold answers saved in {gold_file}")

if __name__ == "__main__":
    main()
