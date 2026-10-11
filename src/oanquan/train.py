"""Vòng lặp AlphaZero: self-play → train → lặp lại. Mỗi thế hệ lưu checkpoint để sau dựng thang Elo.

    python -m oanquan.train --variant S5P5Q10 --gens 40 --games 400 --out runs/standard
    python -m oanquan.train --variant S3P2Q4 --gens 20 --solution results/solved/S3P2Q4.npz --out runs/s3p2

Có --solution thì mỗi thế hệ đo thêm: tỉ lệ nước mạng chọn là nước tối ưu thật, và sai số giá trị.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .net import PolicyValueNet, features_batch
from .rules import Variant, mirror_action
from .selfplay import SelfPlayConfig, worker
from .solver import Solution


def parse_variant(name: str) -> Variant:
    # S5P5Q10 → Variant(5, 5, 10)
    s, rest = name[1:].split("P")
    p, q = rest.split("Q")
    return Variant(int(s), int(p), int(q))


def mirror_tables(v: Variant):
    s, n = v.squares, v.n
    board_idx = np.empty(n + 2, dtype=np.int64)
    board_idx[(s + 1 - np.arange(n)) % n] = np.arange(n)
    board_idx[n], board_idx[n + 1] = n + 1, n
    act_idx = np.array([mirror_action(a, s) for a in range(2 * s)])
    return board_idx, act_idx


class Buffer:
    def __init__(self, keep_gens: int):
        self.keep = keep_gens
        self.parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add(self, b, p, z):
        self.parts.append((b, p, z))
        self.parts = self.parts[-self.keep :]

    def arrays(self):
        b = np.concatenate([x[0] for x in self.parts])
        p = np.concatenate([x[1] for x in self.parts])
        z = np.concatenate([x[2] for x in self.parts])
        return b, p, z


def train_epochs(net, opt, v: Variant, b, p, z, steps: int, batch: int, rng):
    """Mỗi batch lật gương ngẫu nhiên một nửa (bàn cờ đối xứng trái-phải)."""
    board_idx, act_idx = mirror_tables(v)
    net.train()
    tot_p = tot_v = 0.0
    for _ in range(steps):
        idx = rng.integers(0, len(b), batch)
        bb, pp, zz = b[idx].copy(), p[idx].copy(), z[idx]
        flip = rng.random(batch) < 0.5
        bb[flip] = bb[flip][:, board_idx]
        mp_ = pp[flip]
        tmp = np.zeros_like(mp_)
        tmp[:, act_idx] = mp_
        pp[flip] = tmp
        x = torch.from_numpy(features_batch(bb, v.squares, v.stones, v.dan))
        logits, val = net(x)
        target = torch.from_numpy(pp)
        lp = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
        lv = F.mse_loss(val, torch.from_numpy(zz))
        loss = lp + lv * 4.0  # giá trị chuẩn hoá theo tổng điểm nên nhỏ, nhân lên cho cân với loss policy
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot_p += lp.item()
        tot_v += lv.item()
    return tot_p / steps, tot_v / steps


@torch.no_grad()
def eval_vs_solution(net, v: Variant, sol: Solution, idx: np.ndarray, prefix: str, mcts_n: int = 0, sims: int = 160):
    """So mạng với lời giải chính xác trên các thế idx (chỉ thế có giá trị xác định).

    policy_opt: argmax của policy head (không tìm kiếm) có phải nước tối ưu không.
    choice_opt: như trên nhưng chỉ tính thế có nước kém hơn (không phải mọi nước đều tối ưu như nhau).
    mcts_opt: nước MCTS chọn có tối ưu không (đắt nên chỉ chạy trên mcts_n thế đầu).
    value_mae: sai số giá trị mạng so với giá trị thật, tính bằng điểm.
    """
    idx = idx[sol.reliable()[idx]]
    boards = np.stack([sol.board(int(i)) for i in idx])
    x = torch.from_numpy(features_batch(boards, v.squares, v.stones, v.dan))
    net.eval()
    logits, val = net(x)
    legal = sol.succ[idx] != -2
    logits = logits.numpy()
    logits[~legal] = -np.inf
    choice = logits.argmax(1)
    q = sol.q_values(idx)
    best = q.max(1)
    optimal = q[np.arange(len(idx)), choice] == best
    real = (q == best[:, None]).sum(1) < legal.sum(1)
    out = {
        f"{prefix}policy_opt": round(float(optimal.mean()), 4),
        f"{prefix}choice_opt": round(float(optimal[real].mean()), 4),
        f"{prefix}value_mae": round(float(np.abs(val.numpy() * v.total - sol.value[idx]).mean()), 3),
        f"{prefix}n": int(len(idx)),
    }
    if mcts_n:
        from .mcts import search

        w = net.numpy_weights()
        hits = []
        for k in np.flatnonzero(real)[:mcts_n]:
            vis, _, _ = search(boards[k], v.squares, v.stones, v.quan, v.dan, v.total, w, sims, 1.5, 0.3, 0.0)
            hits.append(q[k, int(vis.argmax())] == best[k])
        out[f"{prefix}mcts_choice_opt"] = round(float(np.mean(hits)), 4)
    return out


def lookup_many(sol: Solution, boards: np.ndarray) -> np.ndarray:
    idx = np.array([sol.lookup(b) for b in boards])
    return idx[idx >= 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="S5P5Q10")
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--games", type=int, default=400, help="số ván self-play mỗi thế hệ")
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--keep", type=int, default=6, help="buffer giữ dữ liệu bao nhiêu thế hệ gần nhất")
    ap.add_argument("--steps", type=int, default=400, help="số bước train mỗi thế hệ")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--solution", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="chạy tiếp từ checkpoint mới nhất trong --out")
    ap.add_argument("--temp-plies", type=int, default=12, help="số nước đầu chọn ngẫu nhiên theo lượt thăm")
    ap.add_argument("--qmix", type=float, default=0.0, help="trộn giá trị MCTS ở gốc vào target value")
    args = ap.parse_args()

    v = parse_variant(args.variant)
    out = Path(args.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = PolicyValueNet(v.squares, hidden=args.hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sol = Solution.load(args.solution) if args.solution else None
    if sol is not None:
        uniform_idx = np.random.default_rng(123).choice(len(sol.keys), size=min(20000, len(sol.keys)), replace=False)
    cfg = SelfPlayConfig(sims=args.sims, temp_plies=args.temp_plies, qmix=args.qmix)
    buf = Buffer(args.keep)
    log = open(out / "log.jsonl", "a", encoding="utf-8")

    def save(gen):
        torch.save(net.state_dict(), out / "ckpt" / f"gen{gen:03d}.pt")
        torch.save(opt.state_dict(), out / "ckpt" / "optimizer.pt")

    start = 0
    if args.resume:
        start = max(int(p.stem[3:]) for p in (out / "ckpt").glob("gen*.pt"))
        net.load_state_dict(torch.load(out / "ckpt" / f"gen{start:03d}.pt"))
        if (out / "ckpt" / "optimizer.pt").exists():
            opt.load_state_dict(torch.load(out / "ckpt" / "optimizer.pt"))
            for g in opt.param_groups:
                g["lr"] = args.lr
        # giữ config gốc (kích thước mạng...), cấu hình chạy tiếp lưu riêng
        (out / f"config_resume_gen{start:03d}.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(f"chạy tiếp từ thế hệ {start}", flush=True)
    else:
        (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        save(0)
    if sol is not None and start == 0:
        rec = {"gen": 0, **eval_vs_solution(net, v, sol, uniform_idx, "all_")}
        print(json.dumps(rec, ensure_ascii=False))
        log.write(json.dumps(rec) + "\n")
        log.flush()

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for gen in range(start + 1, start + args.gens + 1):
            t0 = time.perf_counter()
            w = net.numpy_weights()
            per = [args.games // args.workers + (i < args.games % args.workers) for i in range(args.workers)]
            jobs = [(v, w, cfg, k, int(rng.integers(1 << 62))) for k in per if k > 0]
            res = pool.map(worker, jobs)
            b = np.concatenate([r[0] for r in res])
            p = np.concatenate([r[1] for r in res])
            z = np.concatenate([r[2] for r in res])
            margins = np.array([m for r in res for m in r[3]])
            lengths = np.array([x for r in res for x in r[4]])
            buf.add(b, p, z)
            t1 = time.perf_counter()
            lp, lv = train_epochs(net, opt, v, *buf.arrays(), args.steps, args.batch, rng)
            save(gen)
            rec = {
                "gen": gen,
                "selfplay_s": round(t1 - t0, 1),
                "train_s": round(time.perf_counter() - t1, 1),
                "positions": int(len(z)),
                "games": args.games,
                "sims": args.sims,
                "len_mean": float(lengths.mean()),
                "first_player_margin": float(margins.mean()),
                "first_player_win": float((margins > 0).mean()),
                "draw": float((margins == 0).mean()),
                "loss_policy": round(lp, 4),
                "loss_value": round(lv, 5),
            }
            if sol is not None:
                rec.update(eval_vs_solution(net, v, sol, uniform_idx, "all_"))
                # thế thật sự gặp trong self-play thế hệ này (bỏ trùng)
                seen = np.unique(lookup_many(sol, b[rng.choice(len(b), min(len(b), 20000), replace=False)]))
                rec.update(
                    eval_vs_solution(net, v, sol, seen, "play_", mcts_n=300 if gen % 5 == 0 else 0, sims=args.sims)
                )
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
