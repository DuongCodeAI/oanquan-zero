// Luật Ô ăn quan, bản TypeScript của src/oanquan/rules.py. Phải ra đúng từng con số như bản Python
// (có test so với fixtures sinh từ Python: web/src/engine/parity.test.ts).
//
// Vòng N = 2S + 2 ô: 0 = quan trái, 1..S = hàng người 0, S+1 = quan phải, S+2..2S+1 = hàng người 1.
// State: [dân từng ô (N)], quan trái, quan phải, điểm 0, điểm 1, người đi, số nước, đã kết thúc.

export interface Variant {
  squares: number
  stones: number
  quan: number
  maxPly: number
}

export const STANDARD: Variant = { squares: 5, stones: 5, quan: 10, maxPly: 300 }

export const ringSize = (v: Variant) => 2 * v.squares + 2
export const nActions = (v: Variant) => 2 * v.squares
export const totalDan = (v: Variant) => 2 * v.squares * v.stones
export const totalPoints = (v: Variant) => totalDan(v) + 2 * v.quan

const MAX_SOW_STEPS = 20_000

export type State = Int32Array

export function newGame(v: Variant): State {
  const n = ringSize(v)
  const st = new Int32Array(n + 7)
  for (let i = 0; i < v.squares; i++) {
    st[1 + i] = v.stones
    st[v.squares + 2 + i] = v.stones
  }
  st[n] = 1
  st[n + 1] = 1
  return st
}

// các sự kiện để vẽ animation một nước đi
export type SowEvent =
  | { kind: 'spread'; player: number } // hết dân, rải S dân từ kho
  | { kind: 'pick'; pos: number; count: number }
  | { kind: 'drop'; pos: number }
  | { kind: 'capture'; pos: number; dan: number; quan: boolean }
  | { kind: 'stop'; reason: 'quan' | 'two-empty' | 'limit' }
  | { kind: 'end'; reason: 'no-quan' | 'no-stones' | 'max-ply' }

const occupied = (st: State, n: number, s: number, j: number) =>
  st[j] > 0 || (j === 0 && st[n] === 1) || (j === s + 1 && st[n + 1] === 1)

export function rowEmpty(st: State, s: number, player: number): boolean {
  const base = 1 + player * (s + 1)
  for (let i = 0; i < s; i++) if (st[base + i] > 0) return false
  return true
}

export function legalMask(st: State, v: Variant): boolean[] {
  const s = v.squares
  const n = ringSize(v)
  const m = new Array<boolean>(2 * s).fill(false)
  if (st[n + 6] === 1) return m
  const p = st[n + 4]
  const base = 1 + p * (s + 1)
  const empty = rowEmpty(st, s, p)
  for (let i = 0; i < s; i++) {
    if (empty || st[base + i] > 0) m[2 * i] = m[2 * i + 1] = true
  }
  return m
}

function collectRows(st: State, n: number, s: number, p: number): number {
  let diff = 0
  for (let pl = 0; pl < 2; pl++) {
    const b = 1 + pl * (s + 1)
    for (let i = 0; i < s; i++) {
      st[n + 2 + pl] += st[b + i]
      diff += pl === p ? st[b + i] : -st[b + i]
      st[b + i] = 0
    }
  }
  st[n + 6] = 1
  return diff
}

/** Đi nước a tại chỗ. Trả về chênh lệch điểm (người đi − đối thủ) phát sinh trong nước. */
export function step(st: State, a: number, v: Variant, events?: SowEvent[]): number {
  const s = v.squares
  const n = ringSize(v)
  const q = v.quan
  const p = st[n + 4]
  const base = 1 + p * (s + 1)
  let gained = 0

  if (rowEmpty(st, s, p)) {
    for (let i = 0; i < s; i++) st[base + i] = 1
    gained -= s
    events?.push({ kind: 'spread', player: p })
  }

  const d = a % 2 === 0 ? 1 : -1
  let pos = base + Math.floor(a / 2)
  let hand = st[pos]
  st[pos] = 0
  events?.push({ kind: 'pick', pos, count: hand })
  let steps = 0
  for (;;) {
    while (hand > 0) {
      pos = (pos + d + n) % n
      st[pos] += 1
      hand -= 1
      steps += 1
      events?.push({ kind: 'drop', pos })
    }
    const nxt = (pos + d + n) % n
    if (occupied(st, n, s, nxt)) {
      if (nxt === 0 || nxt === s + 1) {
        events?.push({ kind: 'stop', reason: 'quan' })
        break
      }
      if (steps > MAX_SOW_STEPS) {
        events?.push({ kind: 'stop', reason: 'limit' })
        break
      }
      hand = st[nxt]
      st[nxt] = 0
      pos = nxt
      events?.push({ kind: 'pick', pos, count: hand })
      continue
    }
    let tgt = (nxt + d + n) % n
    while (occupied(st, n, s, tgt)) {
      const dan = st[tgt]
      let got = dan
      let quan = false
      st[tgt] = 0
      if (tgt === 0 && st[n] === 1) {
        got += q
        st[n] = 0
        quan = true
      } else if (tgt === s + 1 && st[n + 1] === 1) {
        got += q
        st[n + 1] = 0
        quan = true
      }
      gained += got
      events?.push({ kind: 'capture', pos: tgt, dan, quan })
      const e = (tgt + d + n) % n
      if (occupied(st, n, s, e)) break
      tgt = (e + d + n) % n
    }
    events?.push({ kind: 'stop', reason: 'two-empty' })
    break
  }

  st[n + 2 + p] += gained
  st[n + 4] = 1 - p
  st[n + 5] += 1

  if (st[0] === 0 && st[s + 1] === 0 && st[n] === 0 && st[n + 1] === 0) {
    gained += collectRows(st, n, s, p)
    events?.push({ kind: 'end', reason: 'no-quan' })
  } else if (rowEmpty(st, s, 1 - p)) {
    let onBoard = 0
    for (let j = 0; j < n; j++) onBoard += st[j]
    if (totalDan(v) - onBoard < s) {
      gained += collectRows(st, n, s, p)
      events?.push({ kind: 'end', reason: 'no-stones' })
    }
  }
  if (st[n + 6] === 0 && st[n + 5] >= v.maxPly) {
    st[n + 6] = 1
    events?.push({ kind: 'end', reason: 'max-ply' })
  }
  return gained
}

/** Bàn cờ nhìn từ người đi (N + 2 phần tử: dân từng ô rồi 2 cờ quan). */
export function canonical(st: State, v: Variant): Int32Array {
  const s = v.squares
  const n = ringSize(v)
  const p = st[n + 4]
  const b = new Int32Array(n + 2)
  const shift = p * (s + 1)
  for (let j = 0; j < n; j++) b[j] = st[(j + shift) % n]
  b[n] = p === 1 ? st[n + 1] : st[n]
  b[n + 1] = p === 1 ? st[n] : st[n + 1]
  return b
}

export const isDone = (st: State, v: Variant) => st[ringSize(v) + 6] === 1
export const toMove = (st: State, v: Variant) => st[ringSize(v) + 4]
export const scoreOf = (st: State, v: Variant, p: number) => st[ringSize(v) + 2 + p]
export const plyOf = (st: State, v: Variant) => st[ringSize(v) + 5]
