import { useMemo } from 'react'
import { ringSize, type Variant } from '../engine/rules'

// Bàn vẽ như kẻ phấn trên sân: hai hàng ô dân, hai đầu là ô quan hình bán nguyệt.
const W = 1000
const H = 390
const X0 = 170
const X1 = 830
const YT = 86 // mép trên hàng AI
const YM = 196 // ranh giới hai hàng
const YB = 306 // mép dưới hàng người chơi

export interface Thought {
  visits: number[]
  q: number[]
}

interface Props {
  v: Variant
  st: Int32Array
  hand: { pos: number; count: number } | null
  flash: number | null
  playable: boolean[] // theo action của người chơi (người 0)
  onMove: (a: number) => void
  thought: Thought | null // MCTS của AI (người 1)
  lastAction: { player: number; a: number } | null
  selected: number | null
  onSelect: (i: number | null) => void
}

function cellCenter(idx: number, s: number): [number, number] {
  const w = (X1 - X0) / s
  if (idx === 0) return [X0 - 62, (YT + YB) / 2]
  if (idx === s + 1) return [X1 + 62, (YT + YB) / 2]
  if (idx <= s) return [X0 + (idx - 1 + 0.5) * w, (YM + YB) / 2]
  const k = idx - (s + 2)
  return [X0 + (s - 1 - k + 0.5) * w, (YT + YM) / 2]
}

// sỏi xếp theo xoắn ốc hoa hướng dương, lệch nhẹ theo ô để không ô nào giống ô nào
function pebbleSpots(idx: number, count: number, spread: number) {
  const out: { x: number; y: number; r: number }[] = []
  const shown = Math.min(count, 18)
  const phase = (idx * 2.399) % (2 * Math.PI)
  for (let k = 0; k < shown; k++) {
    const rad = spread * Math.sqrt(k + 0.5)
    const th = k * 2.39996 + phase
    const jitter = Math.sin(idx * 13.1 + k * 7.7) * 0.9
    out.push({ x: rad * Math.cos(th), y: rad * Math.sin(th) * 0.86, r: 10.5 + jitter })
  }
  return out
}

export function Board({ v, st, hand, flash, playable, onMove, thought, lastAction, selected, onSelect }: Props) {
  const s = v.squares
  const n = ringSize(v)
  const w = (X1 - X0) / s

  const totalVisits = useMemo(() => (thought ? thought.visits.reduce((a, b) => a + b, 0) : 0), [thought])
  const canPlay = playable.some(Boolean)

  const quanPath = (left: boolean) => {
    const x = left ? X0 : X1
    const bulge = left ? -112 : 112
    return `M ${x} ${YT} C ${x + bulge} ${YT + 4}, ${x + bulge} ${YB - 4}, ${x} ${YB}`
  }

  const cells = []
  for (let idx = 0; idx < n; idx++) {
    const [cx, cy] = cellCenter(idx, s)
    const isQuan = idx === 0 || idx === s + 1
    const dan = st[idx]
    const hasQuan = isQuan && st[n + (idx === 0 ? 0 : 1)] === 1
    const spots = pebbleSpots(idx, dan, isQuan ? 11 : 11.5)
    const oy = hasQuan ? 30 : 0
    cells.push(
      <g key={idx}>
        {flash === idx && (isQuan ? (
          <ellipse cx={cx} cy={cy} rx={48} ry={100} className="flash" />
        ) : (
          <rect x={cx - w / 2 + 4} y={cy - 51} width={w - 8} height={102} rx={10} className="flash" />
        ))}
        {hasQuan && (
          <g transform={`translate(${cx} ${cy - 30})`}>
            <ellipse rx={24} ry={20} fill="url(#quan)" stroke="var(--gold)" strokeWidth={1.5} />
            <ellipse cx={-7} cy={-7} rx={7} ry={4.5} fill="var(--gold-hi)" opacity={0.7} />
          </g>
        )}
        {spots.map((p, k) => (
          <circle key={k} cx={cx + p.x} cy={cy + oy + p.y} r={p.r} fill="url(#pebble)" stroke="var(--pebble-lo)" strokeWidth={0.8} />
        ))}
        <text
          x={isQuan ? cx : cx + w / 2 - 10}
          y={isQuan ? (idx === 0 ? YB + 22 : YB + 22) : idx <= s ? YB - 10 : YT + 22}
          textAnchor={isQuan ? 'middle' : 'end'}
          className={`count${dan >= 10 ? ' big' : ''}`}
        >
          {dan}
          {hasQuan ? ' + quan' : ''}
        </text>
      </g>,
    )
  }

  // ô người chơi bấm được
  const hits = []
  for (let i = 0; i < s; i++) {
    const idx = 1 + i
    const [cx] = cellCenter(idx, s)
    const ok = playable[2 * i]
    hits.push(
      <rect
        key={idx}
        x={cx - w / 2 + 3}
        y={YM + 3}
        width={w - 6}
        height={YB - YM - 6}
        rx={10}
        className={`cell-hit${ok ? ' playable' : ''}${selected === i ? ' selected' : ''}`}
        onClick={() => ok && onSelect(selected === i ? null : i)}
      />,
    )
  }

  // MCTS của AI: chấm son trên ô, độ lớn theo tỉ lệ lượt thăm
  const thinking = []
  if (thought && totalVisits > 0) {
    for (let i = 0; i < s; i++) {
      const idx = s + 2 + i
      const [cx] = cellCenter(idx, s)
      // với AI (ngồi phía trên), hướng +1 đi từ phải sang trái trên màn hình
      const left = thought.visits[2 * i] / totalVisits
      const right = thought.visits[2 * i + 1] / totalVisits
      for (const [frac, dx, arrow] of [
        [left, -22, '←'],
        [right, 22, '→'],
      ] as const) {
        if (frac < 0.005) continue
        thinking.push(
          <g key={`${idx}${arrow}`}>
            <circle cx={cx + dx} cy={YT - 26} r={3 + 13 * Math.sqrt(frac)} className="think-dot" opacity={0.85} />
            <text x={cx + dx} y={YT - 50} textAnchor="middle" className="think-label">
              {arrow}
              {Math.round(frac * 100)}%
            </text>
          </g>,
        )
      }
    }
  }

  const last = lastAction
    ? (() => {
        const idx = lastAction.player === 0 ? 1 + Math.floor(lastAction.a / 2) : s + 2 + Math.floor(lastAction.a / 2)
        const [cx, cy] = cellCenter(idx, s)
        return <rect x={cx - w / 2 + 6} y={cy - 49} width={w - 12} height={98} rx={9} fill="none" stroke="var(--line)" strokeDasharray="4 5" strokeWidth={1.5} />
      })()
    : null

  let dirButtons = null
  if (selected !== null && playable[2 * selected] && canPlay) {
    const [cx] = cellCenter(1 + selected, s)
    const btn = (a: number, dx: number, label: string, title: string) => (
      <g
        className="dir-btn svg-only"
        role="button"
        tabIndex={0}
        aria-label={title}
        transform={`translate(${cx + dx} ${YB + 40})`}
        onClick={() => {
          onSelect(null)
          onMove(a)
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            onSelect(null)
            onMove(a)
          }
        }}
      >
        <rect x={-30} y={-17} width={60} height={34} rx={17} />
        <text textAnchor="middle" y={6}>
          {label}
        </text>
      </g>
    )
    dirButtons = (
      <>
        {btn(2 * selected + 1, -34, '←', `Rải ô ${selected + 1} sang trái`)}
        {btn(2 * selected, 34, '→', `Rải ô ${selected + 1} sang phải`)}
      </>
    )
  }

  let handMark = null
  if (hand && hand.count > 0) {
    const [cx, cy] = cellCenter(hand.pos, s)
    handMark = (
      <g transform={`translate(${cx} ${cy - 62})`} style={{ transition: 'transform 0.08s linear' }}>
        <rect x={-20} y={-14} width={40} height={24} rx={12} fill="var(--ink)" />
        <text textAnchor="middle" y={3} className="hand" fill="var(--paper)">
          {hand.count}
        </text>
      </g>
    )
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="board-svg" role="img" aria-label="Bàn Ô ăn quan">
      <defs>
        <filter id="chalky" x="-5%" y="-5%" width="110%" height="110%">
          <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="2" seed="4" />
          <feDisplacementMap in="SourceGraphic" scale="4" />
        </filter>
        <radialGradient id="pebble" cx="35%" cy="30%" r="75%">
          <stop offset="0%" stopColor="var(--pebble-hi)" />
          <stop offset="55%" stopColor="var(--pebble)" />
          <stop offset="100%" stopColor="var(--pebble-lo)" />
        </radialGradient>
        <radialGradient id="quan" cx="35%" cy="30%" r="80%">
          <stop offset="0%" stopColor="var(--gold-hi)" />
          <stop offset="100%" stopColor="var(--gold)" />
        </radialGradient>
      </defs>

      <g filter="url(#chalky)">
        <rect x={X0} y={YT} width={X1 - X0} height={YB - YT} className="chalk" />
        <line x1={X0} y1={YM} x2={X1} y2={YM} className="chalk" />
        {Array.from({ length: s - 1 }, (_, i) => (
          <line key={i} x1={X0 + (i + 1) * w} y1={YT} x2={X0 + (i + 1) * w} y2={YB} className="chalk" />
        ))}
        <path d={quanPath(true)} className="chalk" />
        <path d={quanPath(false)} className="chalk" />
      </g>

      <text x={X0} y={YT - 66} className="row-label">
        AI
      </text>
      <text x={X0} y={YB + 66} className="row-label" style={{ fill: 'var(--you)' }}>
        BẠN
      </text>

      {last}
      {hits}
      <g pointerEvents="none">{cells}</g>
      {thinking}
      {dirButtons}
      {handMark}
    </svg>
  )
}
