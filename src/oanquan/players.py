"""Các người chơi để so sánh: ngẫu nhiên, tham ăn, minimax alpha-beta, AlphaZero, và người chơi hoàn hảo
(tra bảng solver, chỉ có cho biến thể nhỏ).

Mọi người chơi nhận bàn canonical (góc nhìn người đi) và trả về action.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .mcts import _child_board, _legal, search
from .rules import Variant, canonical, legal_mask, new_game, step


@njit(cache=True)
def _negamax(b, s, q, dan, depth, alpha, beta):
    n = 2 * s + 2
    m = _legal(b, s)
    # thử nước ăn được nhiều trước để cắt tỉa tốt hơn
    gains = np.full(2 * s, -1e9)
    kids = np.zeros((2 * s, n + 2), dtype=np.int32)
    done = np.zeros(2 * s, dtype=np.bool_)
    st = np.zeros(n + 7, dtype=np.int32)
    for a in range(2 * s):
        if not m[a]:
            continue
        st[:] = 0
        st[: n + 2] = b
        gains[a] = step(st, a, s, q, 1 << 30, dan)
        done[a] = st[n + 6] == 1
        kids[a] = _child_board(st, n, s)
    order = np.argsort(-gains)
    best = -1e9
    best_a = -1
    nodes = 1
    for k in range(2 * s):
        a = order[k]
        if not m[a]:
            continue
        val = gains[a]
        if not done[a] and depth > 1:
            cv, _, cn = _negamax(kids[a], s, q, dan, depth - 1, -beta + gains[a], -alpha + gains[a])
            val -= cv
            nodes += cn
        if val > best:
            best, best_a = val, a
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best, best_a, nodes


class Player:
    name = "?"

    def act(self, b: np.ndarray, rng: np.random.Generator) -> int:
        raise NotImplementedError


class RandomPlayer(Player):
    name = "random"

    def __init__(self, v: Variant):
        self.v = v

    def act(self, b, rng):
        return int(rng.choice(np.flatnonzero(_legal(b, self.v.squares))))


class GreedyPlayer(Player):
    """Đi nước ăn được nhiều điểm nhất ngay (hoà thì chọn ngẫu nhiên)."""

    name = "greedy"

    def __init__(self, v: Variant):
        self.v = v

    def act(self, b, rng):
        return MinimaxPlayer(self.v, 1).act(b, rng)


class MinimaxPlayer(Player):
    def __init__(self, v: Variant, depth: int):
        self.v, self.depth = v, depth
        self.name = f"minimax{depth}"
        self.nodes = 0  # tổng số thế cờ đã xét, để so "chi phí suy nghĩ" với MCTS
        self.moves = 0

    def act(self, b, rng):
        v = self.v
        s, n = v.squares, v.n
        m = _legal(b, s)
        # chấm từng nước ở gốc để chọn ngẫu nhiên giữa các nước bằng điểm
        vals = np.full(2 * s, -np.inf)
        st = np.zeros(n + 7, dtype=np.int32)
        for a in np.flatnonzero(m):
            st[:] = 0
            st[: n + 2] = b
            g = step(st, int(a), s, v.quan, 1 << 30, v.dan)
            vals[a] = g
            self.nodes += 1
            if st[n + 6] == 0 and self.depth > 1:
                cv, _, cn = _negamax(_child_board(st, n, s), s, v.quan, v.dan, self.depth - 1, -1e9, 1e9)
                vals[a] -= cv
                self.nodes += cn
        self.moves += 1
        best = np.flatnonzero(vals == vals.max())
        return int(rng.choice(best))


class ZeroPlayer(Player):
    def __init__(self, v: Variant, weights, sims: int = 200, c_puct: float = 1.5, name: str = "zero"):
        self.v, self.w, self.sims, self.c = v, weights, sims, c_puct
        self.name = name

    def act(self, b, rng):
        v = self.v
        visits, _, _ = search(b, v.squares, v.stones, v.quan, v.dan, v.total, self.w, self.sims, self.c, 0.3, 0.0)
        best = np.flatnonzero(visits == visits.max())
        return int(rng.choice(best))


class PolicyPlayer(Player):
    """Chỉ dùng policy head, không tìm kiếm: đo xem bản thân mạng mạnh tới đâu."""

    def __init__(self, v: Variant, weights, name: str = "policy"):
        self.v, self.w, self.name = v, weights, name

    def act(self, b, rng):
        from .mcts import _evaluate

        p, _, _ = _evaluate(b, self.v.squares, self.v.stones, self.v.dan, self.w)
        return int(np.argmax(p))


class PerfectPlayer(Player):
    name = "perfect"

    def __init__(self, solution):
        self.sol = solution

    def act(self, b, rng):
        idx = self.sol.lookup(b)
        q = self.sol.q_values(idx)
        return int(rng.choice(np.flatnonzero(q == q.max())))


def play_game(v: Variant, p0: Player, p1: Player, rng, opening_random: int = 0):
    """Trả về (điểm người 0 - điểm người 1, số nước). opening_random nước đầu đi ngẫu nhiên để ván đa dạng."""
    st = new_game(v)
    players = (p0, p1)
    n = v.n
    while st[n + 6] == 0:
        b = canonical(st, v.squares)
        if st[n + 5] < opening_random:
            a = int(rng.choice(np.flatnonzero(legal_mask(st, v.squares))))
        else:
            a = players[st[n + 4]].act(b, rng)
        step(st, a, *v.args())
    return int(st[n + 2] - st[n + 3]), int(st[n + 5])


def match(v: Variant, a: Player, b: Player, games: int, seed: int = 0, opening_random: int = 2):
    """Đấu cặp, đổi bên đi trước mỗi ván, hai ván trong một cặp cùng khai cuộc ngẫu nhiên.

    Trả về theo góc nhìn a:
      win/draw/loss, score: tính từng ván.
      pair_win/pair_draw/pair_loss, pair_score: tính theo cặp, cộng chênh lệch điểm của 2 ván. Bàn chuẩn có lợi thế
        đi trước rất lớn (người đi trước thắng ~80% ván self-play) nên thắng/thua từng ván chủ yếu do ai đi trước;
        tính theo cặp mới tách được ai mạnh hơn.
      first_win: tỉ lệ ván người đi trước thắng.
      margin: chênh lệch điểm trung bình mỗi ván.
    """
    rng = np.random.default_rng(seed)
    w = d = lose = 0
    margins, first = [], []
    for g in range(games):
        if g % 2 == 0:
            pair_seed = int(rng.integers(1 << 62))
        game_rng = np.random.default_rng(pair_seed)
        if g % 2 == 0:
            m, _ = play_game(v, a, b, game_rng, opening_random)
            first.append(m)
        else:
            m, _ = play_game(v, b, a, game_rng, opening_random)
            first.append(m)
            m = -m
        margins.append(m)
        w += m > 0
        d += m == 0
        lose += m < 0
    pairs = np.array(margins[: len(margins) // 2 * 2]).reshape(-1, 2).sum(1)
    pw, pd, pl = int((pairs > 0).sum()), int((pairs == 0).sum()), int((pairs < 0).sum())
    return {
        "win": w,
        "draw": d,
        "loss": lose,
        "margin": float(np.mean(margins)),
        "score": (w + 0.5 * d) / games,
        "pair_win": pw,
        "pair_draw": pd,
        "pair_loss": pl,
        "pair_score": (pw + 0.5 * pd) / max(1, len(pairs)),
        "first_win": float(np.mean(np.array(first) > 0)),
    }
