import { useState } from 'react'

export interface Series {
  key: string
  label: string
  color: string // biến CSS
  points: { x: number; y: number }[]
  dashed?: boolean
}

interface LineProps {
  series: Series[]
  yMin: number
  yMax: number
  yTicks: number[]
  yFormat: (y: number) => string
  xLabel: string
  refs?: { y: number; label: string }[]
  height?: number
  caption: string
}

const PAD = { l: 48, r: 96, t: 16, b: 40 }

export function LineChart({ series, yMin, yMax, yTicks, yFormat, xLabel, refs = [], height = 300, caption }: LineProps) {
  const W = 640
  const H = height
  const xs = series.flatMap((s) => s.points.map((p) => p.x))
  const xMin = Math.min(...xs)
  const xMax = Math.max(...xs)
  const sx = (x: number) => PAD.l + ((x - xMin) / (xMax - xMin || 1)) * (W - PAD.l - PAD.r)
  const sy = (y: number) => PAD.t + (1 - (y - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b)
  const [hover, setHover] = useState<number | null>(null)

  const xTicks = (() => {
    const span = xMax - xMin
    const stepX = span > 60 ? 20 : span > 25 ? 10 : 5
    const out = []
    for (let x = Math.ceil(xMin / stepX) * stepX; x <= xMax; x += stepX) out.push(x)
    return out
  })()

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * (W - PAD.l - PAD.r) + PAD.l
    const x = xMin + ((px - PAD.l) / (W - PAD.l - PAD.r)) * (xMax - xMin)
    let best = xs[0]
    for (const v of xs) if (Math.abs(v - x) < Math.abs(best - x)) best = v
    setHover(best)
  }

  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={caption}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={PAD.l - 8} y={sy(t) + 4} textAnchor="end">
              {yFormat(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={sx(t)} y={H - PAD.b + 18} textAnchor="middle">
            {t}
          </text>
        ))}
        <text x={(PAD.l + W - PAD.r) / 2} y={H - 4} textAnchor="middle">
          {xLabel}
        </text>
        <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} className="axis" />

        {refs.map((r) => (
          <g key={r.label}>
            <line x1={PAD.l} x2={W - PAD.r} y1={sy(r.y)} y2={sy(r.y)} stroke="var(--ink-soft)" strokeDasharray="3 4" strokeWidth={1} />
            <text x={W - PAD.r + 6} y={sy(r.y) + 4} className="lbl" style={{ fontSize: 11.5 }}>
              {r.label}
            </text>
          </g>
        ))}

        {series.map((s) => {
          const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ')
          const last = s.points[s.points.length - 1]
          return (
            <g key={s.key}>
              <path d={d} fill="none" stroke={`var(${s.color})`} strokeWidth={2} strokeDasharray={s.dashed ? '5 4' : undefined} strokeLinejoin="round" />
              {s.points.length < 20 &&
                s.points.map((p) => <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r={4} fill={`var(${s.color})`} stroke="var(--paper)" strokeWidth={2} />)}
              <text x={sx(last.x) + 8} y={sy(last.y) + 4} className="lbl">
                {s.label}
              </text>
            </g>
          )
        })}

        {hover !== null && (
          <g pointerEvents="none">
            <line x1={sx(hover)} x2={sx(hover)} y1={PAD.t} y2={H - PAD.b} stroke="var(--ink-soft)" strokeWidth={1} />
            {series.map((s) => {
              const p = s.points.find((q) => q.x === hover)
              return p ? <circle key={s.key} cx={sx(p.x)} cy={sy(p.y)} r={5} fill={`var(${s.color})`} stroke="var(--paper)" strokeWidth={2} /> : null
            })}
          </g>
        )}
        <rect x={PAD.l} y={PAD.t} width={W - PAD.l - PAD.r} height={H - PAD.t - PAD.b} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>
      <div className="legend" aria-live="polite">
        {hover === null
          ? series.map((s) => (
              <span key={s.key}>
                <i style={{ background: `var(${s.color})` }} />
                {s.label}
              </span>
            ))
          : [
              <span key="x">
                {xLabel} {hover}:
              </span>,
              ...series.map((s) => {
                const p = s.points.find((q) => q.x === hover)
                return (
                  <span key={s.key}>
                    <i style={{ background: `var(${s.color})` }} />
                    {s.label} <b style={{ color: 'var(--ink)' }}>{p ? yFormat(p.y) : '–'}</b>
                  </span>
                )
              }),
            ]}
      </div>
    </figure>
  )
}

export interface Compare {
  label: string
  before: number
  after: number
  format: (x: number) => string
  max: number
}

/** Cặp thanh trước/sau cho từng chỉ số cùng đơn vị %. Trước = xám trung tính, sau = màu chính. */
export function BeforeAfter({ rows, beforeLabel, afterLabel }: { rows: Compare[]; beforeLabel: string; afterLabel: string }) {
  return (
    <div className="ba">
      {rows.map((r) => (
        <div key={r.label} className="ba-row">
          <div className="ba-label">{r.label}</div>
          {(
            [
              ['before', r.before, 'var(--ba-before)', beforeLabel],
              ['after', r.after, 'var(--s1)', afterLabel],
            ] as const
          ).map(([k, val, color, name]) => (
            <div key={k} className="ba-bar" title={`${name}: ${r.format(val)}`}>
              <span className="ba-name">{name}</span>
              <span className="ba-track">
                <i style={{ width: `${(val / r.max) * 100}%`, background: color }} />
              </span>
              <b>{r.format(val)}</b>
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

/** Thanh ngang cho một chỉ số 0..1, có vạch 50% (hoà) để thấy ngay bên nào hơn. */
export function ScoreBars({ rows }: { rows: { label: string; value: number; se: number; detail: string }[] }) {
  return (
    <div className="ba">
      {rows.map((r) => (
        <div key={r.label}>
          <div className="ba-label">{r.label}</div>
          <div className="ba-bar" title={r.detail}>
            <span className="ba-name">{r.detail}</span>
            <span className="ba-track score-track">
              <i style={{ width: `${r.value * 100}%`, background: r.value > 0.5 ? 'var(--s1)' : 'var(--ba-before)' }} />
              <em aria-hidden="true" />
            </span>
            <b>
              {Math.round(r.value * 100)}% ±{Math.round(r.se * 100)}
            </b>
          </div>
        </div>
      ))}
      <p className="hint">Vạch giữa = 50%, tức ngang nhau. ± là sai số chuẩn.</p>
    </div>
  )
}
