# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse
import math
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# ----------------------------------------------------------------------------
# Sinusoidal embedding for scalar t ∈ [0,1]
# ----------------------------------------------------------------------------
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
    )
    args = t * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

# ----------------------------------------------------------------------------
# ResNet backbone selector
# ----------------------------------------------------------------------------
class ResNetBackbone(nn.Module):
    def __init__(self, name: str, embed_dim: int):
        super().__init__()
        resnets = {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'resnet152': models.resnet152,
        }
        assert name in resnets, f"Unsupported backbone '{name}'"
        resnet = resnets[name](weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # drop fc
        feat_dim = resnet.fc.in_features
        self.proj = nn.Linear(feat_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        out = self.features(x).view(B, -1)
        return self.proj(out)

# ----------------------------------------------------------------------------
# LabelEncoder: label-embedding vector z_t 
# ----------------------------------------------------------------------------
class LabelEncoder(nn.Module):
    """
    Encodes a label-embedding vector z_t (shape [B, embed_dim]) via a small FC net with skip connection.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.hidden_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, embed_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fc2(self.relu(self.fc1(z)))
        return out + z


class TimeEncoder(nn.Module):
    """
    Encodes a timestamp t (shape [B,1]) into embedding (shape [B, embed_dim]).
    """
    def __init__(self, time_emb_dim: int, embed_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(time_emb_dim, embed_dim),
            nn.ReLU()
        )
        self.time_emb_dim = time_emb_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B,1] -> sinusoidal [B, time_emb_dim]
        te = sinusoidal_embedding(t, self.time_emb_dim)
        return self.fc(te)

# ----------------------------------------------------------------------------
# fuse head to combine image, z, and t features
# ----------------------------------------------------------------------------
class FuseHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.BatchNorm1d(embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )
    @property
    def out_features(self):
        return self.net[-1].out_features
    def forward(self, fx: torch.Tensor, fz: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fx, fz, ft], dim=1)
        return self.net(x)

# ----------------------------------------------------------------------------
# noise schedule module
# ----------------------------------------------------------------------------
class NoiseSchedule(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.gamma_tilde = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, 1), nn.Softplus()
        )
        self.gamma0 = nn.Parameter(torch.tensor(-7.0))
        self.gamma1 = nn.Parameter(torch.tensor(7.0))

    def _gamma_bar(self, t: torch.Tensor) -> torch.Tensor:
        g0 = self.gamma_tilde(torch.zeros_like(t))
        g1 = self.gamma_tilde(torch.ones_like(t))
        return ((self.gamma_tilde(t) - g0) / (g1 - g0 + 1e-8)).clamp(0,1)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        γt = self.gamma0 + (self.gamma1 - self.gamma0) * (1 - self._gamma_bar(t))
        return torch.sigmoid(-γt / 2).clamp(0.01, 0.99)

# ----------------------------------------------------------------------------
# NoPropCT model: combines all components
# ----------------------------------------------------------------------------
class NoPropCT(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        time_emb_dim: int,
        embed_dim: int
    ):
        super().__init__()
        self.backbone       = ResNetBackbone(backbone, embed_dim)
        self.label_enc      = LabelEncoder(embed_dim)
        self.time_enc          = TimeEncoder(time_emb_dim, embed_dim)
        self.fuse           = FuseHead(embed_dim, num_classes)
        self.noise_schedule = NoiseSchedule(hidden_dim=64)
        self.W_embed        = nn.Parameter(torch.zeros(num_classes, embed_dim))

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.noise_schedule.alpha_bar(t)

    def forward_u(self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        fx = self.backbone(x)
        fz = self.label_enc(z_t)
        ft = self.time_enc(t)
        return self.fuse(fx, fz, ft)

# ----------------------------------------------------------------------------
# initialize prototypes in backbone feature space
# ----------------------------------------------------------------------------
from torch.utils.data import DataLoader
import random

def initialize_with_prototypes(
    model: NoPropCT,
    dataset: Dataset,
    num_classes: int,
    device: torch.device,
    samples_per_class: int = 10,
    batch_size: int = 512,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, List[int]]:
    model.eval()

    # 1) Embed entire dataset in batches
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    feats_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            feats = model.backbone(imgs.to(device))
            feats_list.append(feats.cpu())
            labels_list.append(labels)
    all_feats  = torch.cat(feats_list, dim=0)   # [N, D] on CPU
    all_labels = torch.cat(labels_list, dim=0)  # [N]

    D = all_feats.size(1)
    W_proto   = torch.zeros(num_classes, D, device=device)
    proto_idxs = []

    # 2) For each class, randomly pick up to 10 embeddings, then medoid
    for c in range(num_classes):
        # indices of class-c samples
        idxs_c = (all_labels == c).nonzero(as_tuple=True)[0].tolist()
        # randomly choose up to samples_per_class
        chosen = random.sample(idxs_c, min(samples_per_class, len(idxs_c)))
        embs = all_feats[chosen].to(device)     # [≤10, D]
        idxs = torch.tensor(chosen)

        # compute pairwise distances among just these
        dmat = torch.cdist(embs, embs)          # [k, k]
        dmed = dmat.median(dim=1).values        # [k]
        best = torch.argmin(dmed).item()

        W_proto[c]     = embs[best]
        proto_idxs.append(idxs[best].item())

    return W_proto, proto_idxs

# ----------------------------------------------------------------------------
# single training step
# ----------------------------------------------------------------------------
def train_step(model, x, y, optimizer, device, η: float=1.0) -> float:
    B = x.size(0)
    u_y = model.W_embed[y]                         # (B, embed_dim)
    t   = torch.rand(B,1,device=device,requires_grad=True)
    αb  = model.alpha_bar(t)
    snr = αb / (1-αb)
    snr_p = torch.autograd.grad(snr.sum(), t, create_graph=True)[0]
    eps  = torch.randn_like(u_y)
    zt   = αb*u_y + (1-αb).sqrt()*eps
    logits = model.forward_u(x, zt, t)
    p      = F.softmax(logits, dim=1)
    pred_e = p @ model.W_embed
    mse    = F.mse_loss(pred_e, u_y, reduction='none').sum(dim=1, keepdim=True)
    loss_sdm = 0.5 * η * (snr_p * mse).mean()
    loss_kl  = 0.5 * (u_y.pow(2).sum(dim=1)).mean()
    t1 = torch.ones_like(t)
    αb1 = model.alpha_bar(t1)
    z1  = αb1*u_y + (1-αb1).sqrt()*torch.randn_like(u_y)
    loss_ce = F.cross_entropy(model.forward_u(x, z1, t1), y)
    loss = loss_ce + loss_kl + loss_sdm
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()

# ----------------------------------------------------------------------------
# inference (Euler)
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_noprop_ct_inference(model: NoPropCT, x: torch.Tensor, T_steps: int=1000) -> torch.Tensor:
    model.eval()
    B = x.size(0)
    embed_dim = model.W_embed.size(1)
    dt = 1./T_steps
    z  = torch.randn(B, embed_dim, device=x.device)
    for i in range(T_steps):
        t   = torch.full((B,1), i/T_steps, device=x.device)
        αb  = model.alpha_bar(t)
        logits = model.forward_u(x, z, t)
        p      = F.softmax(logits, dim=1)
        pred_e = p @ model.W_embed
        z      = z + dt*(pred_e - z)/(1-αb)
    final_logits = model.forward_u(x, z, torch.ones_like(t))
    return final_logits.argmax(dim=1)

# ----------------------------------------------------------------------------
# inference (Heun)
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_noprop_ct_inference_heun(model: NoPropCT, x: torch.Tensor, T_steps: int=40) -> torch.Tensor:
    model.eval()
    B = x.size(0); embed_dim = model.W_embed.size(1); dt=1./T_steps
    z = torch.randn(B, embed_dim, device=x.device)
    for i in range(T_steps):
        t_n  = torch.full((B,1), i/T_steps, device=x.device)
        t_np1= torch.full((B,1), (i+1)/T_steps, device=x.device)
        αn   = model.alpha_bar(t_n)
        p_n  = F.softmax(model.forward_u(x, z, t_n), dim=1);
        pred_n  = p_n @ model.W_embed; f_n = (pred_n - z)/(1-αn)
        z_mid   = z + dt*f_n
        αm      = model.alpha_bar(t_np1)
        p_mid   = F.softmax(model.forward_u(x, z_mid, t_np1), dim=1)
        pred_mid= p_mid @ model.W_embed; f_mid=(pred_mid-z_mid)/(1-αm)
        z       = z + 0.5*dt*(f_n+f_mid)
    final_logits = model.forward_u(x, z, torch.ones_like(t_n))
    return final_logits.argmax(dim=1)

# ----------------------------------------------------------------------------
# train & eval loop per backbone + dataset
# ----------------------------------------------------------------------------
def train_and_eval(backbone: str, time_emb_dim: int, embed_dim: int, dataset: str, data_root: str, epoches: int):
    print("start")    
    # dataset-specific setup
    if dataset == 'mnist':
        ds_train = torchvision.datasets.MNIST(data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                transforms.Normalize((0.1307,)*3, (0.3081,)*3),
            ]))
        ds_test = torchvision.datasets.MNIST(data_root, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                transforms.Normalize((0.1307,)*3, (0.3081,)*3),
            ]))
        num_classes = 10
    elif dataset == 'cifar10':
        mean, std = (0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)
        ds_train = torchvision.datasets.CIFAR10(data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]))
        ds_test = torchvision.datasets.CIFAR10(data_root, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]))
        num_classes = 10
    elif dataset == 'cifar100':
        mean, std = (0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)
        ds_train = torchvision.datasets.CIFAR100(data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomAffine(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]))
        ds_test = torchvision.datasets.CIFAR100(data_root, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]))
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- {backbone} on {dataset.upper()} ({num_classes} classes) using {device} ---")

    print("dataset config")    
    tr_loader = DataLoader(ds_train, batch_size=2048, shuffle=True, num_workers=8, drop_last=True)
    te_loader = DataLoader(ds_test, batch_size=2048, shuffle=False, num_workers=8)

    # build model
    print("model config")
    model = NoPropCT(backbone, num_classes, time_emb_dim, embed_dim).to(device)
    # initialize W_embed from prototypes
    print("initialize prototypes")
    W_proto, _ = initialize_with_prototypes(model, ds_train, num_classes, device)
    with torch.no_grad():
        model.W_embed.copy_(W_proto)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    print("train start")       
    # training loop
    for ep in range(1, epoches+1):
        t0, total_loss = time.time(), 0.0
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            total_loss += train_step(model, x, y, optimizer, device) * x.size(0)
        avg_loss = total_loss / len(ds_train)
        print(f"Epoch {ep:03d} loss {avg_loss:8.4f} | train {time.time()-t0:3.1f}s", end='')

        if ep % 5 == 0:
            model.eval()
            corr = tot = 0
            eval_t0 = time.time()
            for x, y in te_loader:
                x, y = x.to(device), y.to(device)
                preds = run_noprop_ct_inference_heun(model, x, T_steps=40)
                corr += (preds == y).sum().item()
                tot += y.size(0)
            print(f" | Acc {100*corr/tot:4.2f}% | infer {time.time()-eval_t0:3.1f}s", end='')
        print()

    # final evaluation
    print("Final Heun multi-T eval:")
    for T in [2,5,10,20,40,80,100]:
        ti, corr, tot = time.time(), 0, 0
        for x, y in te_loader:
            x, y = x.to(device), y.to(device)
            preds = run_noprop_ct_inference_heun(model, x, T)
            corr += (preds == y).sum().item(); tot += y.size(0)
        print(f"Heun T={T:3d} acc {100*corr/tot:4.2f}% | infer {time.time()-ti:3.1f}s")

    print("Final Euler multi-T eval:")
    for T in [2,5,10,20,40,80,100]:
        ti, corr, tot = time.time(), 0, 0
        for x, y in te_loader:
            x, y = x.to(device), y.to(device)
            preds = run_noprop_ct_inference(model, x, T)
            corr += (preds == y).sum().item(); tot += y.size(0)
        print(f"Euler T={T:3d} acc {100*corr/tot:4.2f}% | infer {time.time()-ti:3.1f}s")

    # cleanup
    del model, optimizer, ds_train, ds_test, tr_loader, te_loader
    torch.cuda.empty_cache(); gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist','cifar10','cifar100'], required=True)
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--backbone', choices=['resnet18','resnet50','resnet152'], required=True)
    parser.add_argument('--time-emb-dim', type=int, default=64)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--epoches', type=int, default=200)
    args = parser.parse_args()

    print("argparse done", args)
    train_and_eval(
        backbone    = args.backbone,
        time_emb_dim= args.time_emb_dim,
        embed_dim   = args.embed_dim,
        dataset     = args.dataset,
        data_root   = args.data_root,
        epoches     = args.epoches,
    )

if __name__ == '__main__':
    main()