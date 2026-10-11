import numpy as np
import pytest

from oanquan.players import _negamax
from oanquan.rules import Variant
from oanquan.solver import explore, value_iteration


@pytest.mark.parametrize("v", [Variant(2, 2, 4), Variant(2, 3, 6), Variant(3, 1, 2)])
def test_solver_matches_alpha_beta(v):
    """Value iteration hội tụ sau t vòng = giá trị minimax nhìn trước t nước. Hai cách tính độc lập phải khớp."""
    keys, succ, rew = explore(v)
    val, iters, bad = value_iteration(succ, rew)
    assert not bad.any()
    from oanquan.solver import bits_for, decode

    rng = np.random.default_rng(0)
    for i in rng.choice(len(keys), size=min(150, len(keys)), replace=False):
        b = decode(np.int64(keys[i]), v.n, bits_for(v))
        got, _ = _negamax(b, v.squares, v.quan, v.dan, iters + 2, -1e9, 1e9)
        assert got == val[i]


def test_unresolved_states_are_zero_reward_loops():
    """3 × 2: thế không hội tụ = các vòng không ăn quân + vài thế dẫn vào vòng (xem NOTES, lỗi 2)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    v = Variant(3, 2, 4)
    keys, succ, rew = explore(v)
    _, _, bad = value_iteration(succ, rew, max_iters=1200, window=400)
    idx = np.flatnonzero(bad)
    pos = {int(x): k for k, x in enumerate(idx)}
    assert 0 < len(idx) < 200
    edges, rewards = [], []
    for i in idx:
        out = [a for a in range(succ.shape[1]) if int(succ[i, a]) in pos]
        assert out, "thế chưa xác định nào cũng phải đi được vào thế chưa xác định khác"
        for a in out:
            edges.append((pos[int(i)], pos[int(succ[i, a])]))
            rewards.append(rew[i, a])
    e, r = np.array(edges), np.array(rewards)
    m = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(idx), len(idx)))
    _, lab = connected_components(m, directed=True, connection="strong")
    inside = lab[e[:, 0]] == lab[e[:, 1]]
    assert inside.any()
    assert (r[inside] == 0).all(), "trong vòng lặp không được có nước ăn quân"
