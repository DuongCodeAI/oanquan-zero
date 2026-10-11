// MCTS giống src/oanquan/mcts.py: PUCT, phần thưởng trên cạnh, Q chuẩn hoá min-max. Không có nhiễu
// Dirichlet vì trên web chỉ dùng để đấu.

import { features, forward, type Net } from './net'
import { nActions, ringSize, step, totalPoints, type Variant } from './rules'

interface Node {
  board: Int32Array
  prior: Float32Array
  legal: boolean[]
  value: number
  child: (Node | null | 'terminal')[]
  visits: Int32Array
  wsum: Float64Array
  reward: Float64Array
}

export interface SearchResult {
  visits: number[]
  q: number[] // giá trị từng nước, đơn vị điểm, theo góc nhìn người đi
  value: number // mạng ước lượng ở gốc, đơn vị điểm
}

function legalOf(b: Int32Array, s: number): boolean[] {
  let empty = true
  for (let i = 0; i < s; i++) if (b[1 + i] > 0) empty = false
  const m: boolean[] = []
  for (let i = 0; i < s; i++) {
    const ok = empty || b[1 + i] > 0
    m.push(ok, ok)
  }
  return m
}

export function evaluate(net: Net, b: Int32Array, v: Variant) {
  const { logits, value } = forward(net, features(b, v))
  const legal = legalOf(b, v.squares)
  const prior = new Float32Array(logits.length)
  let mx = -Infinity
  for (let a = 0; a < logits.length; a++) if (legal[a] && logits[a] > mx) mx = logits[a]
  let tot = 0
  for (let a = 0; a < logits.length; a++) {
    if (legal[a]) {
      prior[a] = Math.exp(logits[a] - mx)
      tot += prior[a]
    }
  }
  for (let a = 0; a < logits.length; a++) prior[a] /= tot
  return { prior, value, legal }
}

function makeNode(net: Net, board: Int32Array, v: Variant): Node {
  const { prior, value, legal } = evaluate(net, board, v)
  const na = nActions(v)
  return {
    board,
    prior,
    legal,
    value,
    child: new Array(na).fill(null),
    visits: new Int32Array(na),
    wsum: new Float64Array(na),
    reward: new Float64Array(na),
  }
}

function childBoard(st: Int32Array, v: Variant): Int32Array {
  const n = ringSize(v)
  const s = v.squares
  const b = new Int32Array(n + 2)
  for (let j = 0; j < n; j++) b[j] = st[(j + s + 1) % n]
  b[n] = st[n + 1]
  b[n + 1] = st[n]
  return b
}

export function search(net: Net, root: Int32Array, v: Variant, sims: number, cPuct = 1.5): SearchResult {
  const na = nActions(v)
  const n = ringSize(v)
  const total = totalPoints(v)
  const big = { ...v, maxPly: 1 << 30 }
  const rootNode = makeNode(net, root, v)
  let qmin = 1e9
  let qmax = -1e9

  for (let sim = 0; sim < sims; sim++) {
    const path: [Node, number][] = []
    let x = rootNode
    let ret = 0
    for (;;) {
      let ntot = 0
      for (let a = 0; a < na; a++) ntot += x.visits[a]
      const sq = Math.sqrt(ntot + 1)
      let best = -1e30
      let bestA = -1
      for (let a = 0; a < na; a++) {
        if (!x.legal[a]) continue
        const qv = x.visits[a] > 0 ? x.wsum[a] / x.visits[a] : x.value
        const qn = qmax > qmin ? (qv - qmin) / (qmax - qmin) : 0.5
        const sc = qn + (cPuct * x.prior[a] * sq) / (1 + x.visits[a])
        if (sc > best) {
          best = sc
          bestA = a
        }
      }
      const a = bestA
      path.push([x, a])
      const c = x.child[a]
      if (c === null) {
        const st = new Int32Array(n + 7)
        st.set(x.board)
        const g = step(st, a, big)
        x.reward[a] = g / total
        if (st[n + 6] === 1) {
          x.child[a] = 'terminal'
          ret = x.reward[a]
        } else {
          const y = makeNode(net, childBoard(st, v), v)
          x.child[a] = y
          ret = x.reward[a] - y.value
        }
        break
      }
      if (c === 'terminal') {
        ret = x.reward[a]
        break
      }
      if (path.length >= 511) {
        ret = x.reward[a] - c.value
        break
      }
      x = c
    }
    for (let k = path.length - 1; k >= 0; k--) {
      const [node, a] = path[k]
      node.visits[a] += 1
      node.wsum[a] += ret
      const qv = node.wsum[a] / node.visits[a]
      if (qv < qmin) qmin = qv
      if (qv > qmax) qmax = qv
      if (k > 0) {
        const [pn, pa] = path[k - 1]
        ret = pn.reward[pa] - ret
      }
    }
  }
  return {
    visits: Array.from(rootNode.visits),
    q: Array.from(rootNode.visits, (c, a) => (c > 0 ? (rootNode.wsum[a] / c) * total : 0)),
    value: rootNode.value * total,
  }
}
