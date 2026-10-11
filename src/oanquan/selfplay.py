"""Self-play: mạng hiện tại tự đánh với chính nó, mỗi nước lưu (bàn, phân bố lượt thăm MCTS, giá trị thật về sau)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mcts import search
from .rules import Variant, canonical, new_game, step


@dataclass
class SelfPlayConfig:
    sims: int = 160
    c_puct: float = 1.5
    dir_alpha: float = 0.5
    dir_eps: float = 0.25
    temp_plies: int = 12  # mấy nước đầu chọn theo tỉ lệ lượt thăm, sau đó chọn nước thăm nhiều nhất


def play_one(v: Variant, w, cfg: SelfPlayConfig, rng: np.random.Generator):
    s, n = v.squares, v.n
    st = new_game(v)
    boards, pis, gains = [], [], []
    while st[n + 6] == 0:
        b = canonical(st, s)
        visits, _, _ = search(
            b, s, v.stones, v.quan, v.dan, v.total, w, cfg.sims, cfg.c_puct, cfg.dir_alpha, cfg.dir_eps
        )
        pi = visits / visits.sum()
        if st[n + 5] < cfg.temp_plies:
            a = int(rng.choice(len(pi), p=pi))
        else:
            a = int(rng.choice(np.flatnonzero(visits == visits.max())))
        boards.append(b)
        pis.append(pi.astype(np.float32))
        gains.append(step(st, a, *v.args()))
    # giá trị thật của từng thế = chênh lệch điểm người đi kiếm thêm từ đó đến hết ván
    z = np.zeros(len(gains), dtype=np.float32)
    acc = 0.0
    for t in range(len(gains) - 1, -1, -1):
        acc = gains[t] - acc
        z[t] = acc
    margin = int(st[n + 2] - st[n + 3])
    return np.array(boards), np.array(pis), z / v.total, margin


def worker(args):
    v, w, cfg, games, seed = args
    import numba

    numba.set_num_threads(1)
    np.random.seed(seed % (2**32))  # cho Dirichlet noise trong numba
    rng = np.random.default_rng(seed)
    out_b, out_p, out_z, margins, lengths = [], [], [], [], []
    for _ in range(games):
        b, p, z, m = play_one(v, w, cfg, rng)
        out_b.append(b)
        out_p.append(p)
        out_z.append(z)
        margins.append(m)
        lengths.append(len(z))
    return np.concatenate(out_b), np.concatenate(out_p), np.concatenate(out_z), margins, lengths
