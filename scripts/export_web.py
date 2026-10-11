"""Xuất model + số liệu cho web demo.

    python scripts/export_web.py --gens 0 2 5 10 20 40 80

- web/public/models/standard_genXXX.bin: trọng số float32 nối liền (W1 b1 ... Wv bv), manifest.json kèm Elo.
- web/src/data/lab.json: đường học 3 × 2 so với lời giải, bảng biến thể đã giải, Elo, self-play bàn chuẩn.
Mọi số trên web lấy từ file kết quả thật, không gõ tay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oanquan.net import PolicyValueNet

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

BASELINE_LABEL = {
    "random": "đi ngẫu nhiên",
    "greedy": "tham ăn",
    "minimax2": "minimax 2 nước",
    "minimax4": "minimax 4 nước",
    "minimax6": "minimax 6 nước",
    "minimax8": "minimax 8 nước",
}

NOTES = [
    {
        "title": "Luật vay dân làm dân sinh ra từ hư không",
        "body": "Lúc đầu mình cho điểm âm thoải mái khi phải vay để rải. Kết quả: có ván rải 47 lần, bàn cờ nhiều dân "
        "hơn tổng số dân, solver không bao giờ hội tụ. Sửa bằng luật: kho hai bên cộng lại không đủ thì ván dừng.",
    },
    {
        "title": "Đoán sai nguyên nhân ba lần",
        "body": "0,09% thế cờ ở bàn 3 × 2 có giá trị dao động mãi. Mình đoán do hết quan: sai. Đoán do ăn qua ăn lại: "
        "sai. Tìm được một vòng 8 nước không ăn quân thì kết luận luôn; viết test cho cả 78 thế thì test đỏ. Câu đúng: "
        "8 vòng 8 nước không ăn quân, cộng 14 thế dẫn vào vòng. Các thế đó bị loại khỏi phép đo.",
    },
    {
        "title": "Con số 95% đầu tiên bị thổi phồng",
        "body": "Lúc đầu mình chấm AI trên chính các thế nó gặp khi tự đánh: 95% đúng. Nhưng mỗi thế hệ lại được chấm "
        "trên bộ đề khác, và là bộ đề nó quen. Chấm lại trên một bộ cố định lấy từ ván của người chơi hoàn hảo thì "
        "còn 85%. Mọi biểu đồ trên trang dùng bộ cố định.",
    },
    {
        "title": "Ai đi trước thì thắng, nên tỉ lệ thắng từng ván vô dụng",
        "body": "Ở bàn chuẩn người đi trước thắng 77-90% số ván, nên trận nào tính từng ván cũng ra ~50%: "
        "minimax 8 nước đấu minimax 2 nước chỉ 'thắng 69%', nhưng tính theo cặp ván (cùng khai cuộc, đổi bên) "
        "thì thắng 40/40. "
        "Mình đã train thêm một lượt trước khi nhận ra. Giờ mọi số đều tính theo cặp.",
    },
    {
        "title": "40 cặp ván là quá ít để kết luận",
        "body": "Với 40 cặp mình kết luận AI 'chững lại từ thế hệ 40'. Đo lại 100 cặp: thế hệ 100 thắng thế hệ 40 với "
        "0,73 ± 0,04. Kết luận đúng: AI vẫn tiến bộ khi đấu phiên bản cũ của nó, nhưng không tiến bộ khi đấu minimax "
        "8 nước (self-play không bắc cầu).",
    },
]


def export_models(run: Path, gens: list[int], elo: dict[str, float] | None):
    cfg = json.loads((run / "config.json").read_text())
    out = WEB / "public" / "models"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.bin"):
        f.unlink()
    models = []
    for g in gens:
        net = PolicyValueNet(5, hidden=cfg["hidden"])
        net.load_state_dict(torch.load(run / "ckpt" / f"gen{g:03d}.pt"))
        name = f"standard_gen{g:03d}.bin"
        np.concatenate([w.ravel() for w in net.numpy_weights()]).astype("<f4").tofile(out / name)
        models.append({"gen": g, "file": name, "elo": None if elo is None else elo.get(f"gen{g}")})
    manifest = {"variant": cfg["variant"], "hidden": cfg["hidden"], "models": models}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(models)} model → {out}")


def _group(rows, key):
    out = []
    for r in rows:
        if out and key(out[-1][-1]) == key(r):
            out[-1].append(r)
        else:
            out.append([r])
    return out


def read_log(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/standard")
    ap.add_argument("--small", default="runs/S3P2Q4")
    ap.add_argument("--gens", type=int, nargs="+", required=True)
    ap.add_argument("--elo", default="results/elo.json")
    args = ap.parse_args()

    elo_path = ROOT / args.elo
    elo = json.loads(elo_path.read_text()) if elo_path.exists() else None
    export_models(ROOT / args.run, args.gens, elo["elo"] if elo else None)

    summary = json.loads((ROOT / "results/solved/summary.json").read_text())
    small_cfg = json.loads((ROOT / args.small / "config.json").read_text())
    small = summary[small_cfg["variant"]]
    small_log = read_log(ROOT / args.small / "log.jsonl")
    std_log = read_log(ROOT / args.run / "log.jsonl")

    def train_summary(run_dir: Path):
        cfg = json.loads((run_dir / "config.json").read_text())
        log = [r for r in read_log(run_dir / "log.jsonl") if "selfplay_s" in r]
        return {
            "variant": cfg["variant"],
            "gens": len(log),
            "games": sum(r.get("games", cfg["games"]) for r in log),
            "positions": sum(r["positions"] for r in log),
            "sims": log[-1].get("sims", cfg["sims"]),
            "minutes": round(sum(r["selfplay_s"] + r["train_s"] for r in log) / 60, 1),
            "games_per_gen": log[-1].get("games", cfg["games"]),
            "steps": cfg["steps"],
            "batch": cfg["batch"],
            "hidden": cfg["hidden"],
            # các lần train liên tiếp có cùng cấu hình (ván / thế hệ, lượt MCTS)
            "phases": [
                {"gens": len(grp), "games": grp[0].get("games", cfg["games"]), "sims": grp[0].get("sims", cfg["sims"])}
                for grp in _group(log, lambda r: (r.get("games", cfg["games"]), r.get("sims", cfg["sims"])))
            ],
        }

    net = PolicyValueNet(5)
    ev = json.loads((ROOT / "results/eval_small.json").read_text())
    lab = {
        "params": int(sum(p.numel() for p in net.parameters())),
        "train_small": train_summary(ROOT / args.small),
        "train_standard": train_summary(ROOT / args.run),
        "eval_small": ev,
        "learning": {
            "variant": small_cfg["variant"],
            "squares": small["squares"],
            "stones": small["stones"],
            "states": small["states"],
            "exact_root": small["root_value"],
            "sims": small_cfg["sims"],
            "gens": [
                {
                    "gen": r["gen"],
                    "play_policy_opt": r.get("play_choice_opt", r.get("all_choice_opt")),
                    "play_mcts_choice_opt": r.get("play_mcts_choice_opt"),
                    "all_policy_opt": r["all_choice_opt"],
                    "play_value_mae": r.get("play_value_mae"),
                    "first_player_margin": r.get("first_player_margin", 0.0),
                }
                for r in small_log
            ],
        },
        "elo": {
            "points": [],
            "baselines": [],
            "note": "",
        },
        "standard_vs": [],
        "notes": NOTES,
    }
    if elo:
        e = elo["elo"]
        lab["elo"]["points"] = sorted(
            [{"gen": int(k[3:]), "elo": v} for k, v in e.items() if k.startswith("gen")], key=lambda p: p["gen"]
        )
        lab["elo"]["baselines"] = [{"label": BASELINE_LABEL[k], "elo": v} for k, v in e.items() if k in BASELINE_LABEL]
        # trước/sau trên bàn chuẩn: thế hệ 0 và thế hệ cuối đấu cùng các đối thủ viết tay
        zs = sorted(int(k[3:]) for k in e if k.startswith("gen"))
        lo, hi = f"gen{zs[0]}", f"gen{zs[-1]}"

        def score_vs(z, opp):
            for m in elo["matches"]:
                if m["a"] == z and m["b"] == opp:
                    return m["pair_score"]
                if m["a"] == opp and m["b"] == z:
                    return 1 - m["pair_score"]
            return None

        pm_path = ROOT / "results/pair_matches.jsonl"
        pm = read_log(pm_path) if pm_path.exists() else []

        def pair_score(z, opp):
            # ưu tiên trận 100 cặp (pair_match.py), không có thì lấy trận trong giải Elo
            for m in reversed(pm):
                if m["a"] == z and m["b"] == opp:
                    return m["pair_score"], m["pairs"]
            return score_vs(z, opp), 20

        for opp in ["greedy", "minimax2", "minimax4", "minimax6", "minimax8"]:
            (b, nb), (a, na) = pair_score(lo, opp), pair_score(hi, opp)
            if a is not None and b is not None:
                lab["standard_vs"].append(
                    {"opponent": BASELINE_LABEL[opp], "before": b, "after": a, "pairs": min(nb, na)}
                )
    # câu chữ tóm tắt viết tay sau khi xem số, để trong results/web_text.json
    text_path = ROOT / "results/web_text.json"
    if text_path.exists():
        t = json.loads(text_path.read_text(encoding="utf-8"))
        lab["elo"]["note"] = t.get("elo_note", "")
    dst = WEB / "src" / "data"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "lab.json").write_text(json.dumps(lab, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"lab.json: {len(lab['learning']['gens'])} thế hệ nhỏ, {len(std_log)} thế hệ chuẩn")


if __name__ == "__main__":
    main()
