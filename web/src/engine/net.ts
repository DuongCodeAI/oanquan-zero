// Forward của mạng policy/value (MLP 3 lớp ẩn), cùng cách mã hoá bàn cờ như src/oanquan/net.py.

import type { Variant } from './rules'
import { ringSize, totalDan } from './rules'

const ONEHOT = 8

export interface Layer {
  w: Float32Array // (in, out) theo hàng
  b: Float32Array
  nIn: number
  nOut: number
}

export interface Net {
  layers: Layer[] // 3 lớp ẩn + policy + value
}

export function nFeatures(s: number): number {
  return (2 * s + 2) * (ONEHOT + 1) + 5
}

export function features(b: Int32Array, v: Variant): Float32Array {
  const s = v.squares
  const n = ringSize(v)
  const f = new Float32Array(nFeatures(s))
  let onBoard = 0
  for (let j = 0; j < n; j++) {
    const c = b[j]
    onBoard += c
    f[j * (ONEHOT + 1) + Math.min(c, ONEHOT - 1)] = 1
    f[j * (ONEHOT + 1) + ONEHOT] = c / (2 * v.stones)
  }
  const k = n * (ONEHOT + 1)
  f[k] = b[n]
  f[k + 1] = b[n + 1]
  f[k + 2] = onBoard / totalDan(v)
  let mine = 0
  let theirs = 0
  for (let i = 0; i < s; i++) {
    mine += b[1 + i]
    theirs += b[s + 2 + i]
  }
  f[k + 3] = mine === 0 ? 1 : 0
  f[k + 4] = theirs === 0 ? 1 : 0
  return f
}

function dense(x: Float32Array, L: Layer, relu: boolean): Float32Array {
  const out = new Float32Array(L.nOut)
  out.set(L.b)
  for (let i = 0; i < L.nIn; i++) {
    const xi = x[i]
    if (xi === 0) continue
    const row = i * L.nOut
    for (let o = 0; o < L.nOut; o++) out[o] += xi * L.w[row + o]
  }
  if (relu) for (let o = 0; o < L.nOut; o++) if (out[o] < 0) out[o] = 0
  return out
}

export function forward(net: Net, x: Float32Array): { logits: Float32Array; value: number } {
  const [h1, h2, h3, pol, val] = net.layers
  let h = dense(x, h1, true)
  h = dense(h, h2, true)
  h = dense(h, h3, true)
  return { logits: dense(h, pol, false), value: Math.tanh(dense(h, val, false)[0]) }
}

/** Đọc file .bin: các mảng float32 nối liền theo thứ tự W1 b1 W2 b2 W3 b3 Wp bp Wv bv. */
export function loadNet(buf: ArrayBuffer, s: number, hidden: number): Net {
  const all = new Float32Array(buf)
  const shapes: [number, number][] = [
    [nFeatures(s), hidden],
    [hidden, hidden],
    [hidden, hidden],
    [hidden, 2 * s],
    [hidden, 1],
  ]
  let off = 0
  const layers = shapes.map(([nIn, nOut]) => {
    const w = all.subarray(off, off + nIn * nOut)
    off += nIn * nOut
    const b = all.subarray(off, off + nOut)
    off += nOut
    return { w, b, nIn, nOut }
  })
  if (off !== all.length) throw new Error(`file trọng số sai kích thước: ${all.length} != ${off}`)
  return { layers }
}
