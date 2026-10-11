"""Sinh fixtures để test bản TypeScript ra đúng như bản Python (luật, features, forward, MCTS)."""

import json
from pathlib import Path

import numpy as np
import torch

from oanquan.mcts import search
from oanquan.net import PolicyValueNet, features, forward
from oanquan.rules import STANDARD, Variant, canonical, legal_mask, new_game, step

OUT = Path("web/src/engine/fixtures")


def games(v: Variant, count: int, seed: int):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        st = new_game(v)
        rec = {"actions": [], "gains": [], "states": []}
        while st[v.n + 6] == 0:
            a = int(rng.choice(np.flatnonzero(legal_mask(st, v.squares))))
            rec["actions"].append(a)
            rec["gains"].append(int(step(st, a, *v.args())))
            rec["states"].append(st.tolist())
        out.append(rec)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    variants = [STANDARD, Variant(3, 2, 4), Variant(2, 3, 6)]
    rules = {v.name: {"variant": [v.squares, v.stones, v.quan, v.max_ply], "games": games(v, 40, 7)} for v in variants}
    (OUT / "rules.json").write_text(json.dumps(rules), encoding="utf-8")

    v = STANDARD
    torch.manual_seed(3)
    net = PolicyValueNet(v.squares)
    w = net.numpy_weights()
    np.concatenate([x.ravel() for x in w]).astype("<f4").tofile(OUT / "net.bin")
    rng = np.random.default_rng(11)
    cases = []
    for g in games(v, 8, 5):
        for s in g["states"][::4]:
            st = np.array(s, dtype=np.int32)
            if st[v.n + 6]:
                continue
            b = canonical(st, v.squares)
            logits, val = forward(features(b, v.squares, v.stones, v.dan), w)
            visits, q, _ = search(b, v.squares, v.stones, v.quan, v.dan, v.total, w, 64, 1.5, 0.3, 0.0)
            cases.append(
                {"board": b.tolist(), "logits": logits.tolist(), "value": float(val), "visits": visits.tolist()}
            )
    rng.shuffle(cases)
    (OUT / "net.json").write_text(json.dumps({"hidden": 128, "cases": cases[:40]}), encoding="utf-8")
    print(f"{sum(len(x['games']) for x in rules.values())} ván, {min(40, len(cases))} thế cho net/MCTS")


if __name__ == "__main__":
    main()
