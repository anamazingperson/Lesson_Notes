import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

# ========== 数据集模块 ==========
def generate_sin_data(n_points=500):
    x = torch.linspace(-2*torch.pi, 2*torch.pi, n_points).unsqueeze(-1)
    y = torch.sin(x)
    return TensorDataset(x, y)

# ========== 模型 ==========
class FNN(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, n_experts=4, hidden_total=128, top_k=1):
        super().__init__()
        hidden_per_expert = hidden_total // n_experts
        self.n_experts = n_experts
        self.top_k = top_k

        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_per_expert),
                nn.ReLU(),
                nn.Linear(hidden_per_expert, 1)
            ) for _ in range(n_experts)
        ])
        # 门控网络
        self.gate = nn.Linear(1, n_experts)

    def forward(self, x):
        gate_logits = self.gate(x)  # [B, n_experts]
        gate_weights = torch.softmax(gate_logits, dim=-1)

        # Top-k 稀疏选择
        if self.top_k < self.n_experts:
            top_vals, top_idx = torch.topk(gate_weights, self.top_k, dim=-1)
            mask = torch.zeros_like(gate_weights).scatter_(1, top_idx, 1.0)
            gate_weights = gate_weights * mask
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)

        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # [B,1,E]
        y = torch.sum(gate_weights.unsqueeze(1) * expert_outs, dim=-1)
        return y

# ========== 训练器 ==========
class Trainer:
    def __init__(self, model, train_loader, test_loader, lr=0.01):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.opt = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train(self, epochs=200):
        train_losses, test_losses = [], []
        for _ in range(epochs):
            self.model.train()
            for xb, yb in self.train_loader:
                self.opt.zero_grad()
                y_pred = self.model(xb)
                loss = self.loss_fn(y_pred, yb)
                loss.backward()
                self.opt.step()

            # 记录
            train_losses.append(self.evaluate(self.train_loader))
            test_losses.append(self.evaluate(self.test_loader))
        return train_losses, test_losses

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for xb, yb in loader:
                y_pred = self.model(xb)
                total_loss += self.loss_fn(y_pred, yb).item()
        return total_loss / len(loader)

# ========== 实验入口 ==========
dataset = generate_sin_data()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 对比模型
models = {
    "FNN": FNN(hidden=128),
    "MoE(top1)": MoE(n_experts=4, hidden_total=128, top_k=1),
    # "MoE(top2)": MoE(n_experts=4, hidden_total=128, top_k=2),
    # "MoE(all)": MoE(n_experts=4, hidden_total=128, top_k=4),
}

results = {}
for name, model in models.items():
    trainer = Trainer(model, train_loader, test_loader)
    train_losses, test_losses = trainer.train(epochs=300)
    results[name] = (model, train_losses, test_losses)

# 可视化 Loss
plt.figure()
for name, (_, tr, te) in results.items():
    plt.plot(te, label=f"{name} (test)")
plt.legend()
plt.title("Test Loss Comparison")
plt.show()

# 可视化预测
plt.figure(figsize=(10,6))
x_all = torch.linspace(-2*torch.pi, 2*torch.pi, 200).unsqueeze(-1)
y_true = torch.sin(x_all)
plt.plot(x_all.squeeze().numpy(), y_true.squeeze().numpy(), 'k--', label="True sin(x)")
for name, (model, _, _) in results.items():
    with torch.no_grad():
        y_pred = model(x_all)
    plt.plot(x_all.squeeze().numpy(), y_pred.squeeze().numpy(), label=name)
plt.legend()
plt.title("Prediction vs Ground Truth")
plt.show()
