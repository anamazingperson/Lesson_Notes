import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ========== 数据 ==========
x = torch.linspace(-2*torch.pi, 2*torch.pi, 2000).unsqueeze(-1)
y = torch.sin(x)

# ========== 注意力层 ==========
class CustomAttention(nn.Module):
    def __init__(self, d_model=32, num_heads=4, share_type="none"):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.share_type = share_type
        
        # 独立参数
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.W_Q(x).view(B, L, self.num_heads, self.d_head).transpose(1,2)
        K = self.W_K(x).view(B, L, self.num_heads, self.d_head).transpose(1,2)
        V = self.W_V(x).view(B, L, self.num_heads, self.d_head).transpose(1,2)

        # ===== 参数共享策略 =====
        if self.share_type == "kv_share":
            K = K[:,0:1,:].expand_as(K)
            V = V[:,0:1,:].expand_as(V)
        elif self.share_type == "qk_share":
            Q = K
        elif self.share_type == "all_share":
            Q = K = V

        # 注意力计算
        attn = (Q @ K.transpose(-2,-1)) / (self.d_head ** 0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ V
        out = out.transpose(1,2).contiguous().view(B, L, D)
        return self.fc(out), attn

# ========== 小模型 ==========
class TinyTransformer(nn.Module):
    def __init__(self, share_type="none"):
        super().__init__()
        self.embed = nn.Linear(1, 32)
        self.attn = CustomAttention(32, 4, share_type=share_type)
        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        x = self.embed(x)
        x, attn = self.attn(x)
        y = self.mlp(x)
        return y, attn

# ========== 训练函数 ==========
def train_model(share_type, epochs=200):
    model = TinyTransformer(share_type)
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        y_pred, _ = model(x.unsqueeze(0)) # batch=1
        loss = loss_fn(y_pred.squeeze(), y.squeeze())
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses

# 运行对比
modes = ["none", "kv_share", "qk_share", "all_share"]
results = {}
for m in modes:
    model, losses = train_model(m)
    results[m] = (model, losses)

# ========== 可视化 Loss ==========
plt.figure()
for m in modes:
    plt.plot(results[m][1], label=m)
plt.legend()
plt.title("Training Loss")
plt.show()
# ========== 预测可视化 ==========
plt.figure(figsize=(10,6))
plt.plot(x.squeeze().numpy(), y.squeeze().numpy(), 'k--', label="True sin(x)")

for m in modes:
    model, _ = results[m]
    with torch.no_grad():
        y_pred, _ = model(x.unsqueeze(0))
    plt.plot(x.squeeze().numpy(), y_pred.squeeze().numpy(), label=m)

plt.legend()
plt.title("Prediction vs Ground Truth")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
