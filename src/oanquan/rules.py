"""Luật Ô ăn quan, viết bằng numba để self-play và solver chạy đủ nhanh trên CPU.

Bàn cờ là một vòng N = 2S + 2 ô:

    index 0        : ô quan trái
    1 .. S         : hàng của người 0 (trái → phải)
    S + 1          : ô quan phải
    S + 2 .. 2S + 1: hàng của người 1 (phải → trái, tức trái → phải nếu ngồi phía bên kia)

Xoay vòng S + 1 ô thì hàng người 0 thành hàng người 1, nên "góc nhìn người đi" chỉ là một phép xoay.

State là một mảng int32 phẳng (dễ đưa vào numba, dễ copy):
    st[0:N]   số dân trong từng ô (kể cả dân nằm trong ô quan)
    st[N]     ô quan trái còn quan không (0/1)
    st[N+1]   ô quan phải còn quan không
    st[N+2:N+4] điểm hai người (có thể âm khi phải vay dân để rải)
    st[N+4]   người đi
    st[N+5]   số nước đã đi
    st[N+6]   1 nếu ván đã kết thúc

Action a trong [0, 2S): ô thứ a // 2 trong hàng của người đi, hướng +1 nếu a chẵn, -1 nếu a lẻ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

# ván nào rải quá số bước này trong một nước thì cắt (xem NOTES: rải tiếp sức có thể lặp vô hạn)
MAX_SOW_STEPS = 20_000


@dataclass(frozen=True)
class Variant:
    squares: int = 5  # số ô dân mỗi bên
    stones: int = 5  # số dân mỗi ô lúc đầu
    quan: int = 10  # một quan bằng bao nhiêu dân
    max_ply: int = 300  # quá số nước này thì dừng, phần còn trên bàn không tính cho ai

    @property
    def n(self) -> int:
        return 2 * self.squares + 2

    @property
    def n_actions(self) -> int:
        return 2 * self.squares

    @property
    def total(self) -> int:
        return 2 * self.squares * self.stones + 2 * self.quan

    @property
    def name(self) -> str:
        return f"S{self.squares}P{self.stones}Q{self.quan}"

    @property
    def dan(self) -> int:
        return 2 * self.squares * self.stones

    def args(self) -> tuple[int, int, int, int]:
        return self.squares, self.quan, self.max_ply, self.dan


STANDARD = Variant()


def new_game(v: Variant = STANDARD) -> np.ndarray:
    n, s = v.n, v.squares
    st = np.zeros(n + 7, dtype=np.int32)
    st[1 : s + 1] = v.stones
    st[s + 2 : n] = v.stones
    st[n] = st[n + 1] = 1
    return st


@njit(cache=True)
def _occupied(st, n, s, j):
    # ô j có gì không (dân hoặc quan)
    if st[j] > 0:
        return True
    if j == 0:
        return st[n] == 1
    if j == s + 1:
        return st[n + 1] == 1
    return False


@njit(cache=True)
def _take(st, n, s, q, j):
    # ăn toàn bộ ô j, trả về số điểm
    got = st[j]
    st[j] = 0
    if j == 0 and st[n] == 1:
        got += q
        st[n] = 0
    elif j == s + 1 and st[n + 1] == 1:
        got += q
        st[n + 1] = 0
    return got


@njit(cache=True)
def row_empty(st, s, player):
    base = 1 + player * (s + 1)
    for i in range(s):
        if st[base + i] > 0:
            return False
    return True


@njit(cache=True)
def legal_mask(st, s):
    """Mask action hợp lệ. Hàng trống thì mọi ô đều hợp lệ vì sẽ rải trước khi đi."""
    n = 2 * s + 2
    m = np.zeros(2 * s, dtype=np.bool_)
    if st[n + 6] == 1:
        return m
    p = st[n + 4]
    base = 1 + p * (s + 1)
    empty = row_empty(st, s, p)
    for i in range(s):
        if empty or st[base + i] > 0:
            m[2 * i] = True
            m[2 * i + 1] = True
    return m


@njit(cache=True)
def _collect_rows(st, n, s, p):
    # mỗi bên thu dân hàng mình; trả về chênh lệch theo góc nhìn người p
    diff = 0
    for pl in range(2):
        b = 1 + pl * (s + 1)
        for i in range(s):
            st[n + 2 + pl] += st[b + i]
            diff += st[b + i] if pl == p else -st[b + i]
            st[b + i] = 0
    st[n + 6] = 1
    return diff


@njit(cache=True)
def step(st, a, s, q, max_ply, dan):
    """Đi nước a tại chỗ. Trả về chênh lệch điểm (người đi - đối thủ) phát sinh trong nước này.

    dan là tổng số dân của cả ván (2 * S * số dân mỗi ô), cần để biết còn đủ dân để rải không.
    Không kiểm tra hợp lệ, gọi legal_mask trước.
    """
    n = 2 * s + 2
    p = st[n + 4]
    base = 1 + p * (s + 1)
    gained = 0

    # hết dân thì lấy 5 dân (S dân) đã ăn ra rải, không đủ thì vay, cuối ván trả: tính gộp là trừ điểm
    if row_empty(st, s, p):
        for i in range(s):
            st[base + i] = 1
        gained -= s

    d = 1 if a % 2 == 0 else -1
    pos = base + a // 2
    hand = st[pos]
    st[pos] = 0
    steps = 0
    while True:
        while hand > 0:
            pos = (pos + d) % n
            st[pos] += 1
            hand -= 1
            steps += 1
        nxt = (pos + d) % n
        if _occupied(st, n, s, nxt):
            if nxt == 0 or nxt == s + 1:
                break  # gặp ô quan còn quân: mất lượt
            if steps > MAX_SOW_STEPS:
                break
            hand = st[nxt]  # bốc ô kế tiếp rải tiếp
            st[nxt] = 0
            pos = nxt
            continue
        # ô kế trống: ô sau nó có quân thì ăn, rồi xét tiếp (ăn liên tiếp)
        tgt = (nxt + d) % n
        while _occupied(st, n, s, tgt):
            gained += _take(st, n, s, q, tgt)
            e = (tgt + d) % n
            if _occupied(st, n, s, e):
                break
            tgt = (e + d) % n
        break

    st[n + 2 + p] += gained
    st[n + 4] = 1 - p
    st[n + 5] += 1

    # hết quan, tàn dân, thu quân kéo về
    if st[0] == 0 and st[s + 1] == 0 and st[n] == 0 and st[n + 1] == 0:
        gained += _collect_rows(st, n, s, p)
    elif row_empty(st, s, 1 - p):
        # đối thủ hết dân phải rải, nhưng dân ngoài bàn (kho hai bên cộng lại) không đủ S → ván dừng
        on_board = 0
        for j in range(n):
            on_board += st[j]
        if dan - on_board < s:
            gained += _collect_rows(st, n, s, p)
    if st[n + 6] == 0 and st[n + 5] >= max_ply:
        st[n + 6] = 1
    return gained


def canonical(st: np.ndarray, s: int) -> np.ndarray:
    """Bàn cờ nhìn từ phía người đi: hàng người đi luôn là 1..S. Trả về (N + 2,) gồm dân và 2 cờ quan."""
    n = 2 * s + 2
    p = int(st[n + 4])
    pits = np.roll(st[:n], -p * (s + 1))
    quans = st[n : n + 2][::-1] if p == 1 else st[n : n + 2]
    return np.concatenate([pits, quans]).astype(np.int32)


def mirror_board(b: np.ndarray, s: int) -> np.ndarray:
    """Lật trái-phải một bàn canonical: ô j → (S + 1 - j) mod N, đổi hai quan."""
    n = 2 * s + 2
    out = np.empty_like(b)
    idx = (s + 1 - np.arange(n)) % n
    out[idx] = b[:n]
    out[n], out[n + 1] = b[n + 1], b[n]
    return out


def mirror_action(a: int, s: int) -> int:
    i, d = divmod(a, 2)
    return 2 * (s - 1 - i) + (1 - d)


def board_to_state(b: np.ndarray, s: int) -> np.ndarray:
    """Dựng lại state đầy đủ từ bàn canonical (người đi = 0, điểm = 0). Dùng cho solver và test."""
    n = 2 * s + 2
    st = np.zeros(n + 7, dtype=np.int32)
    st[: n + 2] = b[: n + 2]
    return st


def action_name(a: int) -> str:
    i, d = divmod(a, 2)
    return f"ô {i + 1} {'→' if d == 0 else '←'}"
