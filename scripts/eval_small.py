"""Đo trước/sau train trên biến thể có lời giải chính xác, cùng một bộ thế cờ cố định cho mọi thế hệ.

    python scripts/eval_small.py runs/S3P2Q4 --gens 0 1 2 5 10 20 40

Bộ thế "gặp trong ván thật": lấy từ các ván người chơi hoàn hảo tự đánh với nhau, 4 nước đầu ngẫu nhiên để đa
dạng. Không lấy từ self-play của chính AI, vì như vậy mỗi thế hệ được chấm trên bộ đề khác nhau (và dễ hơn).
Ngoài ra đấu 200 ván với người chơi hoàn hảo, tính số điểm AI để mất mỗi ván so với chơi hoàn hảo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oanquan.mcts import search
from oanquan.net import PolicyValueNet, features_batch
from oanquan.players import PerfectPlayer, ZeroPlayer, match
from oanquan.rules import canonical, legal_mask, new_game, step
from oanquan.solver import Solution
from oanquan.train import parse_variant


def perfect_positions(sol: Solution, games: int, seed: int) -> np.ndarray:
    v = sol.variant
    rng = np.random.default_rng(seed)
    pp = PerfectPlayer(sol)
    seen = set()
    for _ in range(games):
        st = new_game(v)
        while st[v.n + 6] == 0 and st[v.n + 5] < 200:
            b = canonical(st, v.squares)
            i = sol.lookup(b)
            seen.add(i)
            if st[v.n + 5] < 4:
                a = int(rng.choice(np.flatnonzero(legal_mask(st, v.squares))))
            else:
                a = pp.act(b, rng)
            step(st, a, *v.args())
    idx = np.array(sorted(seen))
    return idx[sol.reliable()[idx]]


@torch.no_grad()
def score(net, sol: Solution, idx: np.ndarray, sims: int, mcts_n: int):
    v = sol.variant
    boards = np.stack([sol.board(int(i)) for i in idx])
    logits, val = net(torch.from_numpy(features_batch(boards, v.squares, v.stones, v.dan)))
    legal = sol.succ[idx] != -2
    logits = logits.numpy()
    logits[~legal] = -np.inf
    q = sol.q_values(idx)
    best = q.max(1)
    real = (q == best[:, None]).sum(1) < legal.sum(1)
    pol = q[np.arange(len(idx)), logits.argmax(1)] == best
    w = net.numpy_weights()
    hits = []
    for k in np.flatnonzero(real)[:mcts_n]:
        vis, _, _ = search(boards[k], v.squares, v.stones, v.quan, v.dan, v.total, w, sims, 1.5, 0.3, 0.0)
        hits.append(q[k, int(vis.argmax())] == best[k])
    return {
        "policy_opt": float(pol[real].mean()),
        "mcts_opt": float(np.mean(hits)),
        "value_mae": float(np.abs(val.numpy() * v.total - sol.value[idx]).mean()),
        "n_positions": int(real.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gens", type=int, nargs="+", required=True)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--mcts-n", type=int, default=600)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--out", default="results/eval_small.json")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    v = parse_variant(cfg["variant"])
    sol = Solution.load(f"results/solved/{v.name}.npz")
    idx = perfect_positions(sol, 400, seed=7)
    print(f"{len(idx)} thế cờ cố định (có giá trị xác định)", flush=True)
    perfect = PerfectPlayer(sol)
    rows = []
    for g in args.gens:
        net = PolicyValueNet(v.squares, hidden=cfg["hidden"])
        net.load_state_dict(torch.load(run / "ckpt" / f"gen{g:03d}.pt"))
        net.eval()
        r = {"gen": g, **score(net, sol, idx, args.sims, args.mcts_n)}
        # mất bao nhiêu điểm mỗi ván so với chơi hoàn hảo (người hoàn hảo đấu nhau thì chênh 0 theo định nghĩa)
        m = match(v, ZeroPlayer(v, net.numpy_weights(), sims=args.sims), perfect, args.games, seed=11, opening_random=2)
        r.update(vs_perfect_margin=m["margin"], vs_perfect_score=m["score"])
        print(json.dumps(r), flush=True)
        rows.append(r)
    Path(args.out).write_text(
        json.dumps({"variant": v.name, "sims": args.sims, "positions": int(len(idx)), "rows": rows}, indent=1),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
