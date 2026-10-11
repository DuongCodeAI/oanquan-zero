"""MCTS kiểu AlphaZero, nhưng có phần thưởng trên từng cạnh như MuZero.

Ô ăn quan ăn điểm ngay trong lúc đi, nên giá trị của cạnh (x, a) = r(x, a) - V(con), mọi thứ tính theo góc
nhìn người đi ở x và chia cho tổng điểm của ván. Q được chuẩn hoá min-max trong cây (như MuZero) vì độ lớn
của Q thay đổi nhiều theo giai đoạn ván.

Toàn bộ cây là các mảng cấp phát trước, chạy trong numba.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .net import features, forward
from .rules import step

UNSEEN = -2
TERMINAL = -1


@njit(cache=True)
def _child_board(st, n, s):
    b = np.empty(n + 2, dtype=np.int32)
    for j in range(n):
        b[j] = st[(j + s + 1) % n]
    b[n] = st[n + 1]
    b[n + 1] = st[n]
    return b


@njit(cache=True)
def _legal(b, s):
    m = np.zeros(2 * s, dtype=np.bool_)
    empty = True
    for i in range(s):
        if b[1 + i] > 0:
            empty = False
    for i in range(s):
        if empty or b[1 + i] > 0:
            m[2 * i] = True
            m[2 * i + 1] = True
    return m


@njit(cache=True)
def _evaluate(b, s, stones, dan, w):
    logits, v = forward(features(b, s, stones, dan), w)
    m = _legal(b, s)
    p = np.zeros(2 * s, dtype=np.float32)
    mx = -1e30
    for a in range(2 * s):
        if m[a] and logits[a] > mx:
            mx = logits[a]
    tot = 0.0
    for a in range(2 * s):
        if m[a]:
            p[a] = np.exp(logits[a] - mx)
            tot += p[a]
    for a in range(2 * s):
        p[a] /= tot
    return p, v, m


@njit(cache=True)
def search(root, s, stones, q, dan, total, w, sims, c_puct, dir_alpha, dir_eps):
    """Chạy MCTS từ bàn canonical root.

    Trả về (visits[A], q_root[A] theo điểm chuẩn hoá, value của mạng ở gốc).
    dir_eps = 0 thì không thêm nhiễu (dùng khi đấu).
    """
    n = 2 * s + 2
    na = 2 * s
    cap = sims + 2
    boards = np.zeros((cap, n + 2), dtype=np.int32)
    prior = np.zeros((cap, na), dtype=np.float32)
    legal = np.zeros((cap, na), dtype=np.bool_)
    child = np.full((cap, na), UNSEEN, dtype=np.int32)
    visits = np.zeros((cap, na), dtype=np.int32)
    wsum = np.zeros((cap, na), dtype=np.float64)
    reward = np.zeros((cap, na), dtype=np.float64)
    nval = np.zeros(cap, dtype=np.float64)

    boards[0] = root
    p, v, m = _evaluate(root, s, stones, dan, w)
    if dir_eps > 0:
        noise = np.random.gamma(dir_alpha, 1.0, na)
        tot = 0.0
        for a in range(na):
            if m[a]:
                tot += noise[a]
        for a in range(na):
            if m[a]:
                p[a] = (1 - dir_eps) * p[a] + dir_eps * noise[a] / tot
    prior[0] = p
    legal[0] = m
    nval[0] = v
    used = 1
    qmin, qmax = 1e9, -1e9

    path_node = np.zeros(512, dtype=np.int32)
    path_act = np.zeros(512, dtype=np.int32)
    st = np.zeros(n + 7, dtype=np.int32)

    for _ in range(sims):
        x = 0
        depth = 0
        ret = 0.0
        while True:
            # chọn cạnh theo PUCT
            ntot = 0
            for a in range(na):
                ntot += visits[x, a]
            sq = np.sqrt(ntot + 1.0)
            best, best_a = -1e30, -1
            for a in range(na):
                if not legal[x, a]:
                    continue
                if visits[x, a] > 0:
                    qv = wsum[x, a] / visits[x, a]
                else:
                    qv = nval[x]  # chưa thăm: lấy giá trị ước lượng của chính nút này
                qn = (qv - qmin) / (qmax - qmin) if qmax > qmin else 0.5
                sc = qn + c_puct * prior[x, a] * sq / (1 + visits[x, a])
                if sc > best:
                    best, best_a = sc, a
            a = best_a
            path_node[depth] = x
            path_act[depth] = a
            depth += 1
            c = child[x, a]
            if c == UNSEEN:
                st[:] = 0
                st[: n + 2] = boards[x]
                g = step(st, a, s, q, 1 << 30, dan)
                reward[x, a] = g / total
                if st[n + 6] == 1:
                    child[x, a] = TERMINAL
                    ret = reward[x, a]
                else:
                    nb = _child_board(st, n, s)
                    y = used
                    used += 1
                    boards[y] = nb
                    pp, vv, mm = _evaluate(nb, s, stones, dan, w)
                    prior[y] = pp
                    legal[y] = mm
                    nval[y] = vv
                    child[x, a] = y
                    ret = reward[x, a] - vv
                break
            if c == TERMINAL:
                ret = reward[x, a]
                break
            if depth >= 511:
                ret = reward[x, a] - nval[c]
                break
            x = c

        # backup: lên một tầng thì đổi góc nhìn
        for k in range(depth - 1, -1, -1):
            x, a = path_node[k], path_act[k]
            visits[x, a] += 1
            wsum[x, a] += ret
            qv = wsum[x, a] / visits[x, a]
            if qv < qmin:
                qmin = qv
            if qv > qmax:
                qmax = qv
            if k > 0:
                ret = reward[path_node[k - 1], path_act[k - 1]] - ret

    qroot = np.zeros(na, dtype=np.float64)
    for a in range(na):
        if visits[0, a] > 0:
            qroot[a] = wsum[0, a] / visits[0, a]
    return visits[0].copy(), qroot, nval[0]
