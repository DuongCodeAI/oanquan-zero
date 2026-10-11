"""Thống kê ván đánh ngẫu nhiên: độ dài ván, tỉ lệ cắt ở max_ply, lợi thế đi trước, tốc độ engine."""

import sys
import time

import numpy as np

from oanquan.rules import STANDARD, Variant, legal_mask, new_game, step


def run(v: Variant, games: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n, s = v.n, v.squares
    lengths, margins, capped, spreads = [], [], 0, 0
    t = time.perf_counter()
    moves = 0
    for _ in range(games):
        st = new_game(v)
        while st[n + 6] == 0:
            m = legal_mask(st, s)
            p = st[n + 4]
            base = 1 + p * (s + 1)
            if st[base : base + s].sum() == 0:
                spreads += 1
            step(st, int(rng.choice(np.flatnonzero(m))), *v.args())
            moves += 1
        lengths.append(int(st[n + 5]))
        margins.append(int(st[n + 2] - st[n + 3]))
        capped += int(st[n + 5] >= v.max_ply)
    dt = time.perf_counter() - t
    lengths, margins = np.array(lengths), np.array(margins)
    print(
        f"{v.name}: {games} ván, dài TB {lengths.mean():.1f} (p50 {np.median(lengths):.0f}, max {lengths.max()}), "
        f"cắt max_ply {capped}, rải/ván {spreads / games:.2f}, "
        f"người đi trước thắng {np.mean(margins > 0):.3f} hoà {np.mean(margins == 0):.3f} "
        f"chênh TB {margins.mean():+.2f}, {moves / dt / 1000:.0f}k nước/s"
    )


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    for v in [STANDARD, Variant(3, 3, 6), Variant(3, 2, 4), Variant(2, 3, 6), Variant(4, 2, 4)]:
        run(v, games)
