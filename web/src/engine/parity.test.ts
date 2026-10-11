import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import rulesFx from './fixtures/rules.json'
import netFx from './fixtures/net.json'
import { search } from './mcts'
import { features, forward, loadNet } from './net'
import { newGame, step, type Variant } from './rules'

describe('luật khớp bản Python', () => {
  for (const [name, fx] of Object.entries(rulesFx)) {
    it(name, () => {
      const [squares, stones, quan, maxPly] = fx.variant
      const v: Variant = { squares, stones, quan, maxPly }
      for (const g of fx.games) {
        const st = newGame(v)
        g.actions.forEach((a, i) => {
          expect(step(st, a, v)).toBe(g.gains[i])
          expect(Array.from(st)).toEqual(g.states[i])
        })
      }
    })
  }
})

describe('mạng và MCTS khớp bản Python', () => {
  const v: Variant = { squares: 5, stones: 5, quan: 10, maxPly: 300 }
  const raw = readFileSync(new URL('./fixtures/net.bin', import.meta.url))
  const net = loadNet(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength), 5, netFx.hidden)

  it('forward', () => {
    for (const c of netFx.cases) {
      const { logits, value } = forward(net, features(Int32Array.from(c.board), v))
      c.logits.forEach((l, i) => expect(logits[i]).toBeCloseTo(l, 4))
      expect(value).toBeCloseTo(c.value, 5)
    }
  })

  it('MCTS 64 lượt cho cùng số lượt thăm', () => {
    let same = 0
    for (const c of netFx.cases) {
      const r = search(net, Int32Array.from(c.board), v, 64)
      if (r.visits.join() === c.visits.join()) same++
    }
    // float32 cộng theo thứ tự khác nhau có thể đổi một lần chọn khi hai nước sát điểm; yêu cầu gần như toàn bộ khớp
    expect(same).toBeGreaterThanOrEqual(netFx.cases.length - 1)
  })
})
