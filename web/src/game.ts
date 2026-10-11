// Tải các thế hệ model và điều phối ván chơi giữa người và AI.

import { search, type SearchResult } from './engine/mcts'
import { loadNet, type Net } from './engine/net'
import { canonical, STANDARD } from './engine/rules'

export const V = STANDARD

export interface ModelInfo {
  gen: number
  file: string
  elo: number | null
}

export interface Manifest {
  variant: string
  hidden: number
  models: ModelInfo[]
}

const cache = new Map<string, Net>()

export async function loadManifest(): Promise<Manifest> {
  const r = await fetch('/models/manifest.json')
  if (!r.ok) throw new Error(`không tải được danh sách model (${r.status})`)
  return r.json()
}

export async function loadModel(m: Manifest, info: ModelInfo): Promise<Net> {
  const hit = cache.get(info.file)
  if (hit) return hit
  const r = await fetch(`/models/${info.file}`)
  if (!r.ok) throw new Error(`không tải được ${info.file} (${r.status})`)
  const net = loadNet(await r.arrayBuffer(), V.squares, m.hidden)
  cache.set(info.file, net)
  return net
}

export function think(net: Net, st: Int32Array, sims: number): SearchResult {
  return search(net, canonical(st, V), V, Math.max(1, sims))
}

export function pickBest(r: SearchResult): number {
  let best = -1
  let bestA = 0
  r.visits.forEach((c, a) => {
    if (c > best) {
      best = c
      bestA = a
    }
  })
  return bestA
}

export const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms))
