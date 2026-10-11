"""Giải chính xác các biến thể nhỏ, lưu bảng vào results/solved/ và bảng tóm tắt results/solved/summary.json."""

import json
import sys
from pathlib import Path

import numpy as np

from oanquan.rules import Variant
from oanquan.solver import Solution, solve

VARIANTS = [
    Variant(2, 1, 2),
    Variant(2, 2, 4),
    Variant(2, 3, 6),
    Variant(2, 4, 8),
    Variant(3, 1, 2),
    Variant(3, 2, 4),
    Variant(3, 3, 6),
    Variant(4, 1, 2),
    Variant(4, 2, 4),
    Variant(5, 1, 2),
]

if __name__ == "__main__":
    out = Path("results/solved")
    out.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:]
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for v in VARIANTS:
        if names and v.name not in names:
            continue
        path = out / f"{v.name}.npz"
        if path.exists():
            sol = Solution.load(str(path))
        else:
            try:
                sol = solve(v)
            except RuntimeError as e:
                print(e, flush=True)
                continue
            sol.save(str(path))
        q0 = sol.q_values(0)
        summary[v.name] = {
            "squares": v.squares,
            "stones": v.stones,
            "quan": v.quan,
            "total_points": v.total,
            "states": int(len(sol.keys)),
            "unresolved": int(sol.unresolved.sum()),
            "root_value": int(sol.value[0]),
            "root_unresolved": bool(sol.unresolved[0]),
            "first_moves": {int(a): (None if not np.isfinite(q) else int(q)) for a, q in enumerate(q0)},
        }
        summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
