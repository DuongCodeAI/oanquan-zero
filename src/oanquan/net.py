"""Mạng policy/value và cách mã hoá bàn cờ.

Mạng nhỏ (MLP ~50k tham số) vì phải chạy được MCTS vài trăm lần mô phỏng mỗi nước trên CPU và trên trình
duyệt. Forward được viết lại bằng numba (cho MCTS) và bằng TypeScript (cho web); cả ba bản phải ra cùng số.
"""

from __future__ import annotations

import numpy as np
import torch
from numba import njit
from torch import nn

ONEHOT = 8  # đếm 0..6 riêng, từ 7 trở lên gộp chung


def n_features(s: int) -> int:
    return (2 * s + 2) * (ONEHOT + 1) + 5


@njit(cache=True)
def features(b, s, stones, dan):
    """b: bàn canonical (N + 2). Trả về vector float32."""
    n = 2 * s + 2
    f = np.zeros(n * (ONEHOT + 1) + 5, dtype=np.float32)
    on_board = 0
    for j in range(n):
        c = b[j]
        on_board += c
        f[j * (ONEHOT + 1) + min(c, ONEHOT - 1)] = 1.0
        f[j * (ONEHOT + 1) + ONEHOT] = c / (2.0 * stones)
    k = n * (ONEHOT + 1)
    f[k] = b[n]
    f[k + 1] = b[n + 1]
    f[k + 2] = on_board / dan
    mine = 0
    theirs = 0
    for i in range(s):
        mine += b[1 + i]
        theirs += b[s + 2 + i]
    f[k + 3] = 1.0 if mine == 0 else 0.0
    f[k + 4] = 1.0 if theirs == 0 else 0.0
    return f


class PolicyValueNet(nn.Module):
    def __init__(self, s: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers, d = [], n_features(s)
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.policy = nn.Linear(hidden, 2 * s)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.policy(h), torch.tanh(self.value(h)).squeeze(-1)

    def numpy_weights(self) -> tuple[np.ndarray, ...]:
        """(W1, b1, ..., Wp, bp, Wv, bv), W dạng (in, out) float32 contiguous cho numba / web."""
        out = []
        for m in [*[m for m in self.body if isinstance(m, nn.Linear)], self.policy, self.value]:
            out.append(np.ascontiguousarray(m.weight.detach().numpy().T, dtype=np.float32))
            out.append(np.ascontiguousarray(m.bias.detach().numpy(), dtype=np.float32))
        return tuple(out)


@njit(cache=True)
def forward(x, w):
    """Forward bằng numba. w = tuple weight như numpy_weights(), 3 lớp ẩn."""
    h = np.maximum(x @ w[0] + w[1], np.float32(0.0))
    h = np.maximum(h @ w[2] + w[3], np.float32(0.0))
    h = np.maximum(h @ w[4] + w[5], np.float32(0.0))
    logits = h @ w[6] + w[7]
    v = np.tanh((h @ w[8] + w[9])[0])
    return logits, v


@njit(cache=True)
def features_batch(boards, s, stones, dan):
    out = np.zeros((len(boards), n_features_nb(s)), dtype=np.float32)
    for i in range(len(boards)):
        out[i] = features(boards[i], s, stones, dan)
    return out


@njit(cache=True)
def n_features_nb(s):
    return (2 * s + 2) * (ONEHOT + 1) + 5
