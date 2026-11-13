"""CPU-sized didactic adaptation of karpathy/nanoGPT's model.py.

This is a *teaching implementation*, not a verbatim distribution of upstream model.py.
Upstream: https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py
MIT; original upstream copyright 2022 Andrej Karpathy.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 40
    block_size: int = 32
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 32
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self):
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd) < 1:
            raise ValueError('model dimensions must be positive')
        if self.n_embd % self.n_head:
            raise ValueError('n_embd must be divisible by n_head')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')

class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        reshape = lambda t: t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=LayerNorm(cfg.n_embd, cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f'context {T} exceeds block_size {self.config.block_size}')
        if idx.dtype != torch.long:
            raise ValueError('token IDs must be torch.long')
        position = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(position))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x if targets is not None else x[:, [-1], :])
        loss = (F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                ignore_index=-1) if targets is not None else None)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        if temperature <= 0:
            raise ValueError('temperature must be > 0')
        self.eval()
        for _ in range(max_new_tokens):
            cond = idx[:, -self.config.block_size:]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = max(1, min(top_k, logits.size(-1)))
                threshold = torch.topk(logits, k).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, -float('inf'))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def training_text(repeats=60):
    # Original tiny synthetic text, not Shakespeare or copyrighted corpus.
    verses = [
        'the moon follows the quiet sea.\n',
        'small models learn one token at a time.\n',
        'a tiny transformer reads from left to right.\n',
        'data becomes tokens; tokens become predictions.\n',
        'we train, we test, we plot, and we learn.\n',
    ]
    return ''.join(verses * repeats)


def char_dataset(text=None):
    text = text if text is not None else training_text()
    chars = sorted(set(text))
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}
    ids = torch.tensor([stoi[s] for s in text], dtype=torch.long)
    return ids, stoi, itos


def get_batch(ids, block_size, batch_size=8, generator=None):
    if len(ids) <= block_size:
        raise ValueError('text must be longer than block_size')
    at = torch.randint(0, len(ids) - block_size, (batch_size,), generator=generator)
    x = torch.stack([ids[i:i+block_size] for i in at])
    y = torch.stack([ids[i+1:i+block_size+1] for i in at])
    return x, y


def lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr):
    if total_steps <= warmup_steps or warmup_steps < 0:
        raise ValueError('total_steps must be > warmup_steps >= 0')
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    ratio = min(1., max(0., (step - warmup_steps) / (total_steps - warmup_steps)))
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def parameter_groups(model, weight_decay=0.1):
    params = [p for p in model.parameters() if p.requires_grad]
    return [{'params': [p for p in params if p.dim() >= 2], 'weight_decay': weight_decay},
            {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0}]
