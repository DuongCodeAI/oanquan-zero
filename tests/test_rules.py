import numpy as np
import pytest

from oanquan.rules import (
    STANDARD,
    Variant,
    board_to_state,
    canonical,
    legal_mask,
    mirror_action,
    mirror_board,
    new_game,
    step,
)

S, Q = STANDARD.squares, STANDARD.quan
N = STANDARD.n


def total_points(st, v=STANDARD):
    n = v.n
    return int(st[:n].sum() + v.quan * (st[n] + st[n + 1]) + st[n + 2] + st[n + 3])


def play(st, a, v=STANDARD):
    return step(st, a, *v.args())


def board(pits, quans=(1, 1)):
    st = np.zeros(N + 7, dtype=np.int32)
    st[:N] = pits
    st[N], st[N + 1] = quans
    return st


def test_new_game():
    st = new_game()
    assert st[:N].tolist() == [0, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5]
    assert total_points(st) == 70
    assert legal_mask(st, S).all()


def test_sow_then_stop_at_quan():
    # ô 5 có 1 dân, rải sang phải vào ô quan phải; ô kế là ô 7 (hàng đối thủ) có dân → bốc rải tiếp
    st = board([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    st[1] = 1  # để hàng không trống, tránh rải
    g = play(st, 2 * 4 + 0)
    # dân rơi vào ô quan phải (6), ô kế (7) trống, ô 8 trống → mất lượt
    assert g == 0
    assert st[6] == 1 and st[5] == 0


def test_capture_and_chain():
    # ô 1 có 1 dân, đi sang phải: rơi vào ô 2, ô 3 trống, ô 4 có 3 → ăn 3; ô 5 trống, ô 6 (quan) có quân → ăn tiếp
    st = board([0, 1, 0, 0, 3, 0, 2, 0, 0, 0, 0, 0])
    g = play(st, 0)
    assert g == 3 + 2 + Q
    assert st[4] == 0 and st[6] == 0 and st[N + 1] == 0
    assert st[N + 2] == g and st[N + 4] == 1


def test_relay_pickup():
    # ô 1 có 1 dân → rơi vào ô 2; ô 3 có 2 dân → bốc, rải vào 4, 5; ô 6 là quan → mất lượt
    st = board([0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    g = play(st, 0)
    assert g == 0
    assert st[:7].tolist() == [0, 0, 1, 0, 1, 1, 0]


def test_cannot_pick_up_quan_square():
    # rải hết ngay trước ô quan còn dân (không còn quan) → vẫn mất lượt, không bốc ô quan
    st = board([0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0], quans=(1, 0))
    g = play(st, 2 * 3 + 0)
    assert g == 0 and st[6] == 4 and st[5] == 1


def test_empty_quan_square_counts_as_empty():
    # dân cuối rơi vào ô 5, ô quan phải trống hẳn, ô 7 có 2 → ăn 2
    st = board([0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0], quans=(1, 0))
    g = play(st, 2 * 3 + 0)
    assert g == 2


def test_spread_when_row_empty_costs_points():
    st = board([3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    st[N + 2] = 10
    m = legal_mask(st, S)
    assert m.all()
    before = total_points(st)
    play(st, 0)
    assert total_points(st) == before


def test_game_end_collects_rows():
    # ăn nốt quan cuối cùng: ván kết thúc, mỗi bên thu dân hàng mình
    st = board([0, 1, 0, 0, 0, 0, 3, 2, 0, 0, 0, 4], quans=(0, 0))
    g = play(st, 0)  # ô 1 → ô 2, ô 3 trống, ô 4 trống → mất lượt
    assert st[N + 6] == 0
    st = board([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], quans=(1, 0))
    st[3] = 2
    st[8] = 4
    # ô 5 (1 dân) ← : rơi vào ô 4? không, ô 5 là index 5. dùng ô 4 (index 4) sang trái → rơi vào 3
    g = play(st, 2 * 3 + 1)
    # rơi vào index 3 (thành 3 dân), ô kế index 2 trống, index 1 trống → mất lượt
    assert g == 0 and st[N + 6] == 0
    # thế kết thúc thật: chỉ còn quan trái, người 0 ăn được
    st = board([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3], quans=(1, 0))
    st[5] = 2
    g = play(st, 2 * 1 + 1)  # index 2 ← : rơi vào 1, ô 0... ô kế là ô quan trái còn quan → mất lượt
    assert g == 0
    st = board([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], quans=(1, 0))
    st[5] = 2
    g = play(st, 2 * 0 + 0)  # index 1 → rơi vào 2, ô 3 trống, ô 4 trống → mất lượt
    assert g == 0
    st = board([0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 3], quans=(1, 0))
    # index 2 sang phải: rơi vào 3, ô 4 trống, ô 5 có 2 → ăn 2; ô 6 trống, ô 7 trống → dừng
    g = play(st, 2 * 1 + 0)
    assert g == 2 and st[N + 6] == 0


def test_end_when_both_quan_squares_empty():
    st = board([0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0], quans=(1, 0))
    st[11] = 0
    st[0] = 0
    # index 2 sang trái: rơi vào 1, ô 0 (quan trái) trống? không, còn quan → mất lượt
    play(st, 2 * 1 + 1)
    assert st[N + 6] == 0
    st = board([0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], quans=(1, 0))
    st[2] = 1
    st[3] = 1
    st[N + 4] = 0
    # index 3 sang trái: rơi vào 2 (thành 2)? ô 2 có 1 → sau khi rải ô 2 có 2. ô kế (1) trống, ô 0 có quan → ăn quan
    g = play(st, 2 * 2 + 1)
    assert g >= Q
    assert st[N + 6] == 1  # hết cả hai quan → kết thúc, thu quân
    assert st[:N].sum() == 0
    assert total_points(st) == Q + 4


@pytest.mark.parametrize("v", [STANDARD, Variant(3, 2, 4), Variant(2, 3, 6), Variant(4, 1, 2)])
def test_random_games_conserve_and_end(v):
    rng = np.random.default_rng(0)
    for _ in range(300):
        st = new_game(v)
        while st[v.n + 6] == 0:
            legal = np.flatnonzero(legal_mask(st, v.squares))
            assert len(legal) > 0
            step(st, int(rng.choice(legal)), *v.args())
            assert total_points(st, v) == v.total
            assert (st[: v.n] >= 0).all()


@pytest.mark.parametrize("v", [STANDARD, Variant(3, 2, 4)])
def test_mirror_equivariance(v):
    """Lật bàn rồi đi nước lật phải ra đúng bàn lật của kết quả gốc."""
    rng = np.random.default_rng(1)
    s, n = v.squares, v.n
    for _ in range(200):
        st = new_game(v)
        for _ in range(int(rng.integers(0, 30))):
            legal = np.flatnonzero(legal_mask(st, s))
            if st[n + 6] or len(legal) == 0:
                break
            step(st, int(rng.choice(legal)), *v.args())
        if st[n + 6]:
            continue
        b = canonical(st, s)
        for a in np.flatnonzero(legal_mask(board_to_state(b, s), s)):
            x = board_to_state(b, s)
            gx = step(x, int(a), s, v.quan, 10**6, v.dan)
            y = board_to_state(mirror_board(b, s), s)
            gy = step(y, mirror_action(int(a), s), s, v.quan, 10**6, v.dan)
            assert gx == gy
            assert (mirror_board(canonical(x, s), s) == canonical(y, s)).all()


def test_canonical_rotation():
    st = new_game()
    st[2] = 9
    st[N + 4] = 1
    b = canonical(st, S)
    # nhìn từ người 1: hàng của người 1 nằm ở 1..5, ô 2 cũ của người 0 thành index 2 + 6 = 8
    assert b[8] == 9


@pytest.mark.parametrize("v", [STANDARD, Variant(2, 3, 6), Variant(3, 2, 4)])
def test_board_never_exceeds_physical_stones(v):
    """Vay dân không được đẻ ra dân mới: trên bàn không bao giờ quá tổng số dân của ván."""
    rng = np.random.default_rng(2)
    for _ in range(500):
        st = new_game(v)
        while st[v.n + 6] == 0:
            step(st, int(rng.choice(np.flatnonzero(legal_mask(st, v.squares)))), *v.args())
            assert st[: v.n].sum() <= v.dan
