# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import math
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# ----------------------------------------------------------------------------
# Sinusoidal embedding for scalar t in [0,1]
# ----------------------------------------------------------------------------
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1))
    args = t * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

# ----------------------------------------------------------------------------
# ResNet backbone selector
# ----------------------------------------------------------------------------
class ResNetBackbone(nn.Module):
    def __init__(self, name: str, embed_dim: int = 256):
        super().__init__()
        resnets = {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'resnet152': models.resnet152,
        }
        assert name in resnets, f"Unsupported backbone '{name}'"
        resnet = resnets[name](weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # remove fc
        feat_dim = resnet.fc.in_features
        self.proj = nn.Linear(feat_dim, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        out = self.features(x).view(B, -1)
        return self.proj(out)

# ----------------------------------------------------------------------------
# z_t encoder
# ----------------------------------------------------------------------------
class ZEncoder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.ReLU()
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

# ----------------------------------------------------------------------------
# time embedding encoder
# ----------------------------------------------------------------------------
class TEncoder(nn.Module):
    def __init__(self, time_emb_dim: int = 64, embed_dim: int = 256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(time_emb_dim, embed_dim),
            nn.ReLU()
        )
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        te = sinusoidal_embedding(t, self.time_emb_dim)
        return self.net(te)

# ----------------------------------------------------------------------------
# fuse head to combine image, z, and t features
# ----------------------------------------------------------------------------
class FuseHead(nn.Module):
    def __init__(self, embed_dim: int = 256, mid_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.BatchNorm1d(embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, mid_dim),
            nn.BatchNorm1d(mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, num_classes)
        )
    def forward(self, fx: torch.Tensor, fz: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([fx, fz, ft], dim=1)
        return self.net(cat)
    def __getitem__(self, idx):
        return self.net[idx]

# ----------------------------------------------------------------------------
# noise schedule module for learnable gamma and alpha_bar
# ----------------------------------------------------------------------------
class NoiseSchedule(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.gamma_tilde = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, 1), nn.Softplus()
        )
        self.gamma0 = nn.Parameter(torch.tensor(-5.0))
        self.gamma1 = nn.Parameter(torch.tensor(5.0))
    def _gamma_bar(self, t: torch.Tensor) -> torch.Tensor:
        g0 = self.gamma_tilde(torch.zeros_like(t))
        g1 = self.gamma_tilde(torch.ones_like(t))
        return ((self.gamma_tilde(t) - g0) / (g1 - g0 + 1e-8)).clamp(0,1)
    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        γt = self.gamma0 + (self.gamma1 - self.gamma0) * (1 - self._gamma_bar(t))
        return torch.sigmoid(-γt).clamp(1e-5, 1-1e-5)

# ----------------------------------------------------------------------------
# NoPropCT model integrating all components
# ----------------------------------------------------------------------------
class NoPropCT(nn.Module):
    def __init__(self,
                 backbone: str = 'resnet18',
                 num_classes: int = 10,
                 time_emb_dim: int = 64,
                 embed_dim: int = 256):
        super().__init__()
        self.backbone = ResNetBackbone(backbone, embed_dim)
        self.z_enc = ZEncoder(num_classes, embed_dim)
        self.t_enc = TEncoder(time_emb_dim, embed_dim)
        self.fuse = FuseHead(embed_dim, mid_dim=128, num_classes=num_classes)
        self.noise = NoiseSchedule(hidden_dim=64)
    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.noise.alpha_bar(t)
    def forward_u(self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        fx = self.backbone(x)
        fz = self.z_enc(z_t)
        ft = self.t_enc(t)
        return self.fuse(fx, fz, ft)

# ----------------------------------------------------------------------------
# single training step
# ----------------------------------------------------------------------------
def train_step(model, x, y, optimizer, device, η: float = 1.0) -> float:
    B = x.size(0)
    m = model.fuse[-1].out_features
    u_y = torch.eye(m, device=device)[y]
    t = torch.rand(B,1,device=device,requires_grad=True)
    αb = model.alpha_bar(t)
    snr = αb / (1 - αb)
    snr_p = torch.autograd.grad(snr.sum(), t, create_graph=True)[0]
    eps = torch.randn_like(u_y)
    zt = αb * u_y + torch.sqrt(1 - αb) * eps
    logits = model.forward_u(x, zt, t)
    p = F.softmax(logits, dim=1)
    mse = F.mse_loss(p, u_y, reduction='none').sum(dim=1, keepdim=True)
    loss_sdm = 0.5 * η * (snr_p * mse).mean()
    loss_kl = 0.5 * (u_y.pow(2).sum(dim=1)).mean()
    t1 = torch.ones_like(t)
    αb1 = model.alpha_bar(t1)
    z1 = αb1 * u_y + torch.sqrt(1 - αb1) * torch.randn_like(u_y)
    loss_ce = F.cross_entropy(model.forward_u(x, z1, t1), y)
    loss = loss_ce + loss_kl + loss_sdm
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ----------------------------------------------------------------------------
# inference routines
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_noprop_ct_inference(model: nn.Module, x: torch.Tensor, T_steps: int = 1000) -> torch.Tensor:
    model.eval()
    B = x.size(0)
    m = model.fuse[-1].out_features
    dt = 1.0 / T_steps
    z = torch.randn(B, m, device=x.device)
    for i in range(T_steps):
        t = torch.full((B,1), float(i) / T_steps, device=x.device)
        αb = model.alpha_bar(t)
        p = F.softmax(model.forward_u(x, z, t), dim=1)
        z = z + dt * (p - z) / (1 - αb)
    return z.argmax(dim=1)

# ----------------------------------------------------------------------------
# inference routines with heun
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_noprop_ct_inference_heun(model: nn.Module, x: torch.Tensor, T_steps: int = 40) -> torch.Tensor:
    model.eval()
    B = x.size(0)
    m = model.fuse[-1].out_features
    dt = 1.0 / T_steps
    z = torch.randn(B, m, device=x.device)
    for i in range(T_steps):
        t_n = torch.full((B,1), float(i) / T_steps, device=x.device)
        t_np1 = torch.full((B,1), float(i+1) / T_steps, device=x.device)
        αn = model.alpha_bar(t_n)
        p_n = F.softmax(model.forward_u(x, z, t_n), dim=1)
        f_n = (p_n - z) / (1 - αn)
        z_mid = z + dt * f_n
        αm = model.alpha_bar(t_np1)
        p_mid = F.softmax(model.forward_u(x, z_mid, t_np1), dim=1)
        f_mid = (p_mid - z_mid) / (1 - αm)
        z = z + 0.5 * dt * (f_n + f_mid)
    return z.argmax(dim=1)

# ----------------------------------------------------------------------------
# train & eval loop per backbone
# ----------------------------------------------------------------------------
def train_and_eval(backbone: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n--- {backbone} on {device} ---")

    # data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize((0.1307,)*3, (0.3081,)*3),
    ])
    ds_train = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    ds_test = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
    tr = DataLoader(ds_train, batch_size=2048, shuffle=True, num_workers=8, drop_last=True)
    te = DataLoader(ds_test,  batch_size=2048, shuffle=False, num_workers=8)

    # model and optimizer
    model = NoPropCT(backbone, num_classes=10, time_emb_dim=64, embed_dim=256).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # training epochs
    for ep in range(1, 201):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            loss = train_step(model, x, y, optim, device)
            total_loss += loss * x.size(0)
        train_time = time.time() - t0
        avg_loss = total_loss / len(ds_train)
        print(f"Epoch {ep:02d} loss {avg_loss:.4f} | train {train_time:.1f}s", end='')

        # quick eval every 5 epochs
        if ep % 5 == 0:
            ti = time.time()
            model.eval()
            corr = tot = 0
            for x, y in te:
                x, y = x.to(device), y.to(device)
                preds = run_noprop_ct_inference_heun(model, x)
                corr += (preds == y).sum().item(); tot += y.size(0)
            inf_time = time.time() - ti
            print(f" | Acc {corr/tot*100:.2f}% | infer {inf_time:.1f}s", end='')
        print()

    # final multi-T Heun
    print("\nFinal Heun multi-T eval:")
    for T in [2,5,10,20,30,40,50,60,70,80,90,100,200]:
        ti = time.time()
        corr = tot = 0
        for x, y in te:
            x, y = x.to(device), y.to(device)
            preds = run_noprop_ct_inference_heun(model, x, T)
            corr += (preds == y).sum().item(); tot += y.size(0)
        print(f"Heun T={T:3d} acc {corr/tot:.4%} | infer {time.time()-ti:.1f}s")

    # final multi-T Euler
    print("\nFinal Euler multi-T eval:")
    for T in [2,5,10,20,30,40,50,60,70,80,90,100,200]:
        ti = time.time()
        corr = tot = 0
        for x, y in te:
            x, y = x.to(device), y.to(device)
            preds = run_noprop_ct_inference(model, x, T)
            corr += (preds == y).sum().item(); tot += y.size(0)
        print(f"Euler T={T:3d} acc {corr/tot:.4%} | infer {time.time()-ti:.1f}s")

    # cleanup
    del model, optim, tr, te, ds_train, ds_test
    torch.cuda.empty_cache(); gc.collect()

def main():
    for backbone in ['resnet18', 'resnet50', 'resnet152']:
        train_and_eval(backbone)

if __name__ == '__main__':
    main()