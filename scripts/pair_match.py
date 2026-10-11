"""Đấu nhiều cặp ván giữa hai người chơi để có kết quả đủ chắc (song song nhiều tiến trình).

python scripts/pair_match.py runs/standard gen100 gen40 --pairs 100
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from elo import make_player

from oanquan.players import match
from oanquan.train import parse_variant


def job(args):
    a, b, variant, run, sims, games, seed = args
    import numba

    numba.set_num_threads(1)
    v = parse_variant(variant)
    return match(v, make_player(a, v, Path(run), sims), make_player(b, v, Path(run), sims), games, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="results/pair_matches.jsonl")
    args = ap.parse_args()
    variant = json.loads((Path(args.run) / "config.json").read_text())["variant"]
    per = args.pairs // args.workers
    rng = np.random.default_rng(hash((args.a, args.b)) % (2**32))
    jobs = [
        (args.a, args.b, variant, args.run, args.sims, 2 * per, int(rng.integers(1 << 62))) for _ in range(args.workers)
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(job, jobs)
    pw = sum(r["pair_win"] for r in res)
    pd = sum(r["pair_draw"] for r in res)
    pl = sum(r["pair_loss"] for r in res)
    n = pw + pd + pl
    s = (pw + 0.5 * pd) / n
    # sai số chuẩn của điểm cặp (xấp xỉ nhị thức, hoà tính nửa)
    se = np.sqrt(max(s * (1 - s), 1e-9) / n)
    margin = float(np.mean([r["margin"] for r in res]))
    rec = {
        "run": args.run,
        "a": args.a,
        "b": args.b,
        "sims": args.sims,
        "pairs": n,
        "pair_win": pw,
        "pair_draw": pd,
        "pair_loss": pl,
        "pair_score": s,
        "se": float(se),
        "margin": margin,
    }
    print(json.dumps(rec, ensure_ascii=False))
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
