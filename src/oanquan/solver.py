"""Giải chính xác Ô ăn quan cho các biến thể nhỏ.

Giá trị một thế cờ = (điểm người đi kiếm thêm) - (điểm đối thủ kiếm thêm) từ đây đến hết ván, khi cả hai
chơi tối ưu để tối đa hoá chênh lệch. Giá trị này chỉ phụ thuộc bàn cờ, không phụ thuộc điểm đã ăn, vì:
  - ăn quân chỉ cộng điểm cho người ăn;
  - rải khi hết dân luôn tốn đúng S điểm (thiếu thì vay, cuối ván trả), nên không cần biết đang có bao nhiêu.

Ván có thể lặp (đi mãi không ăn được gì). Luật gốc không nói gì về chuyện này; quy ước ở đây giống chỗ cắt
max_ply khi self-play: ván không kết thúc thì phần quân còn trên bàn không tính cho ai. Value iteration bắt đầu
từ 0 chính là giá trị của ván bị cắt sau t nước; khi nó hội tụ (không còn thay đổi) thì đó là điểm bất động.

Bước 1 duyệt hết thế cờ đến được (BFS, numba), bước 2 lặp Bellman bằng numpy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from .rules import Variant, new_game, step


def bits_for(v: Variant) -> int:
    most = 2 * v.squares * v.stones
    return int(most).bit_length()


@njit(cache=True)
def encode(b, n, bits):
    key = np.int64(0)
    for j in range(n):
        key = (key << bits) | b[j]
    key = (key << 1) | b[n]
    key = (key << 1) | b[n + 1]
    return key


@njit(cache=True)
def decode(key, n, bits):
    b = np.zeros(n + 2, dtype=np.int32)
    mask = (np.int64(1) << bits) - 1
    b[n + 1] = key & 1
    key >>= 1
    b[n] = key & 1
    key >>= 1
    for j in range(n - 1, -1, -1):
        b[j] = key & mask
        key >>= bits
    return b


@njit(cache=True)
def _next_board(st, n, s):
    # state sau khi đi: người đi mới là người 1 trong hệ cũ → xoay về góc nhìn của họ
    b = np.empty(n + 2, dtype=np.int32)
    for j in range(n):
        b[j] = st[(j + s + 1) % n]
    b[n] = st[n + 1]
    b[n + 1] = st[n]
    return b


@njit(cache=True)
def _explore(root, s, q, dan, bits, limit):
    n = 2 * s + 2
    na = 2 * s
    index = Dict.empty(key_type=types.int64, value_type=types.int64)
    keys = List()
    succ = List()
    rew = List()
    k0 = encode(root, n, bits)
    index[k0] = 0
    keys.append(k0)
    head = 0
    while head < len(keys):
        b = decode(keys[head], n, bits)
        head += 1
        empty = True
        for i in range(s):
            if b[1 + i] > 0:
                empty = False
        for a in range(na):
            if not empty and b[1 + a // 2] == 0:
                succ.append(np.int64(-2))  # không hợp lệ
                rew.append(np.int32(0))
                continue
            st = np.zeros(n + 7, dtype=np.int32)
            st[: n + 2] = b
            g = step(st, a, s, q, 1 << 30, dan)
            rew.append(np.int32(g))
            if st[n + 6] == 1:
                succ.append(np.int64(-1))  # ván kết thúc
                continue
            nb = _next_board(st, n, s)
            k = encode(nb, n, bits)
            if k in index:
                succ.append(index[k])
            else:
                idx = len(keys)
                index[k] = idx
                keys.append(k)
                succ.append(np.int64(idx))
        if len(keys) > limit:
            return keys, succ, rew, False
    return keys, succ, rew, True


@dataclass
class Solution:
    variant: Variant
    keys: np.ndarray  # (M,) int64, keys[0] là thế khởi đầu
    succ: np.ndarray  # (M, A): -2 không hợp lệ, -1 kết thúc, còn lại là index thế sau
    reward: np.ndarray  # (M, A) chênh lệch điểm người đi kiếm được ngay trong nước đó
    value: np.ndarray  # (M,) giá trị tối ưu
    iters: int
    unresolved: np.ndarray  # (M,) bool, thế trong chu trình có ăn quân, không có giá trị xác định

    @property
    def bits(self) -> int:
        return bits_for(self.variant)

    def q_values(self, idx: np.ndarray | int) -> np.ndarray:
        """Giá trị từng action (−inf nếu không hợp lệ)."""
        s, r = self.succ[idx], self.reward[idx]
        q = r - np.where(s >= 0, self.value[np.maximum(s, 0)], 0)
        return np.where(s == -2, -np.inf, q).astype(np.float64)

    def lookup(self, board: np.ndarray) -> int:
        k = int(encode(board.astype(np.int32), self.variant.n, self.bits))
        if not hasattr(self, "_pos"):
            self._pos = {int(x): i for i, x in enumerate(self.keys)}
        return self._pos.get(k, -1)

    def board(self, idx: int) -> np.ndarray:
        return decode(np.int64(self.keys[idx]), self.variant.n, self.bits)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            keys=self.keys,
            succ=self.succ,
            reward=self.reward,
            value=self.value,
            unresolved=self.unresolved,
            meta=np.array([self.variant.squares, self.variant.stones, self.variant.quan, self.iters]),
        )

    @staticmethod
    def load(path: str) -> Solution:
        z = np.load(path)
        s, p, q, iters = (int(x) for x in z["meta"])
        return Solution(Variant(s, p, q), z["keys"], z["succ"], z["reward"], z["value"], iters, z["unresolved"])

    def reliable(self) -> np.ndarray:
        """Thế có giá trị xác định và mọi thế con cũng vậy (để so nước đi tối ưu không bị nhiễu)."""
        kid_bad = np.where(self.succ >= 0, self.unresolved[np.maximum(self.succ, 0)], False).any(1)
        return ~self.unresolved & ~kid_bad


def explore(v: Variant, limit: int = 60_000_000):
    root = new_game(v)[: v.n + 2]
    keys, succ, rew, done = _explore(root, v.squares, v.quan, v.dan, bits_for(v), limit)
    if not done:
        raise RuntimeError(f"{v.name}: quá {limit} thế cờ, bỏ")
    na = v.n_actions
    keys = np.asarray(keys, dtype=np.int64)
    succ = np.asarray(succ, dtype=np.int64).reshape(-1, na)
    rew = np.asarray(rew, dtype=np.int32).reshape(-1, na)
    idx_dtype = np.int32 if len(keys) < 2**31 else np.int64
    return keys, succ.astype(idx_dtype), rew


def value_iteration(
    succ: np.ndarray, reward: np.ndarray, max_iters: int = 3000, window: int = 1000, log_every: int = 0
):
    """Lặp Bellman từ 0. Trả về (value, số vòng, mask thế chưa hội tụ).

    Một số ít thế nằm trong chu trình có ăn quân (ăn qua ăn lại rồi rải lại mãi): giá trị ván bị cắt sau t nước
    dao động theo t, không có giới hạn. Những thế còn đổi giá trị trong `window` vòng cuối bị đánh dấu là chưa
    xác định và loại khỏi mọi phép đo.
    """
    legal = succ != -2
    nxt = np.maximum(succ, 0)
    term = succ < 0
    v = np.zeros(len(succ), dtype=np.int32)
    last_change = np.zeros(len(succ), dtype=np.int32)
    neg = np.int32(-(1 << 20))
    for it in range(1, max_iters + 1):
        q = reward - np.where(term, 0, v[nxt])
        q = np.where(legal, q, neg)
        new = q.max(axis=1)
        diff = new != v
        last_change[diff] = it
        v = new
        changed = int(diff.sum())
        if log_every and it % log_every == 0:
            print(f"  iter {it}: {changed} thế đổi giá trị")
        if changed == 0:
            return v, it, np.zeros(len(v), dtype=bool)
    return v, max_iters, last_change > max_iters - window


def solve(v: Variant, verbose: bool = True) -> Solution:
    t = time.perf_counter()
    keys, succ, rew = explore(v)
    t1 = time.perf_counter()
    val, iters, bad = value_iteration(succ, rew)
    t2 = time.perf_counter()
    if verbose:
        print(
            f"{v.name}: {len(keys):,} thế cờ, duyệt {t1 - t:.1f}s, value iteration {iters} vòng {t2 - t1:.1f}s, "
            f"chưa xác định {int(bad.sum())}, giá trị thế đầu {val[0]:+d}{' (CHƯA XÁC ĐỊNH)' if bad[0] else ''}",
            flush=True,
        )
    return Solution(v, keys, succ, rew, val, iters, bad)
