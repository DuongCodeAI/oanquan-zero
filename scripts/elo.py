"""Thang Elo: các thế hệ AlphaZero + đối thủ viết tay đấu với nhau, Elo ước lượng bằng MLE (Bradley-Terry).

    python scripts/elo.py runs/standard --gens 0 1 2 5 10 20 40 80 --games 40

Mỗi cặp người chơi đấu `games` ván = games/2 "cặp ván": cùng 2 nước khai cuộc ngẫu nhiên, đổi bên đi trước.
Elo tính trên kết quả **cặp ván** (cộng chênh lệch điểm của 2 ván), không phải từng ván: bàn chuẩn có lợi thế đi
trước rất lớn nên thắng/thua từng ván chủ yếu do ai đi trước (xem NOTES). Mốc 0 = người đi ngẫu nhiên.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from oanquan.net import PolicyValueNet
from oanquan.players import GreedyPlayer, MinimaxPlayer, RandomPlayer, ZeroPlayer, match
from oanquan.train import parse_variant


def make_player(spec: str, v, run: Path, sims: int):
    if spec == "random":
        return RandomPlayer(v)
    if spec == "greedy":
        return GreedyPlayer(v)
    if spec.startswith("minimax"):
        return MinimaxPlayer(v, int(spec[7:]))
    # "gen30@runs/standard_v2" = thế hệ 30 của một lần chạy khác (để so A/B)
    if "@" in spec:
        spec, other = spec.split("@")
        run = Path(other)
    gen = int(spec[3:])
    cfg = json.loads((run / "config.json").read_text())
    net = PolicyValueNet(v.squares, hidden=cfg["hidden"])
    net.load_state_dict(torch.load(run / "ckpt" / f"gen{gen:03d}.pt"))
    return ZeroPlayer(v, net.numpy_weights(), sims=sims, name=spec)


def run_pair(args):
    a, b, variant, run, sims, games, seed = args
    import numba

    numba.set_num_threads(1)
    v = parse_variant(variant)
    pa, pb = make_player(a, v, Path(run), sims), make_player(b, v, Path(run), sims)
    t = time.perf_counter()
    r = match(v, pa, pb, games, seed=seed)
    r.update(a=a, b=b, seconds=round(time.perf_counter() - t, 1))
    return r


def fit_elo(players: list[str], results: list[dict], anchor: str = "random") -> dict[str, float]:
    """MLE Bradley-Terry trên kết quả cặp ván (thắng 1, hoà 0.5). Prior Gauss sigma 1000 Elo để cặp thắng tuyệt đối
    không đẩy Elo ra vô hạn."""
    from scipy.optimize import minimize

    idx = {p: i for i, p in enumerate(players)}
    k = np.log(10) / 400
    ia = np.array([idx[m["a"]] for m in results])
    ja = np.array([idx[m["b"]] for m in results])
    S = np.array([m["pair_win"] + 0.5 * m["pair_draw"] for m in results], dtype=float)
    N = np.array([m["pair_win"] + m["pair_draw"] + m["pair_loss"] for m in results], dtype=float)

    def nll(r):
        d = k * (r[ia] - r[ja])
        # log sigmoid ổn định số
        ll = S * -np.logaddexp(0, -d) + (N - S) * -np.logaddexp(0, d)
        e = 1 / (1 + np.exp(-d))
        g = np.zeros_like(r)
        np.add.at(g, ia, -k * (S - N * e))
        np.add.at(g, ja, k * (S - N * e))
        return -ll.sum() + (r**2).sum() / (2 * 1000**2), g + r / 1000**2

    r = minimize(nll, np.zeros(len(players)), jac=True, method="L-BFGS-B").x
    r -= r[idx[anchor]]
    return {p: float(r[idx[p]]) for p in players}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gens", type=int, nargs="+", required=True)
    ap.add_argument(
        "--baselines", nargs="+", default=["random", "greedy", "minimax2", "minimax4", "minimax6", "minimax8"]
    )
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/elo.json")
    args = ap.parse_args()

    run = Path(args.run)
    variant = json.loads((run / "config.json").read_text())["variant"]
    zeros = [f"gen{g}" for g in args.gens]
    players = args.baselines + zeros
    # baseline đấu nhau hết; mỗi thế hệ đấu mọi baseline và 2 thế hệ kề hai bên
    pairs = list(itertools.combinations(args.baselines, 2))
    pairs += [(z, b) for z in zeros for b in args.baselines]
    pairs += [(zeros[i], zeros[j]) for i in range(len(zeros)) for j in range(i + 1, min(len(zeros), i + 3))]
    rng = np.random.default_rng(0)
    jobs = [(a, b, variant, str(run), args.sims, args.games, int(rng.integers(1 << 62))) for a, b in pairs]
    print(f"{len(players)} người chơi, {len(jobs)} cặp, {len(jobs) * args.games} ván", flush=True)
    t = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = []
        for r in pool.imap_unordered(run_pair, jobs):
            results.append(r)
            wdl = f"cặp {r['pair_win']:>2}-{r['pair_draw']}-{r['pair_loss']:<2}"
            print(
                f"  {r['a']:>9} vs {r['b']:<9} {wdl} cách biệt {r['margin']:+.1f}  ({r['seconds']}s)",
                flush=True,
            )
    elo = fit_elo(players, results)
    for p in sorted(players, key=elo.get):
        print(f"{p:>10} {elo[p]:7.0f}")
    print(f"xong sau {time.perf_counter() - t:.0f}s")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"variant": variant, "sims": args.sims, "games_per_pair": args.games, "elo": elo, "matches": results},
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
