import { useCallback, useEffect, useRef, useState } from 'react'
import { evaluate } from '../engine/mcts'
import type { Net } from '../engine/net'
import { canonical, isDone, legalMask, newGame, plyOf, ringSize, scoreOf, step, toMove, totalPoints, type SowEvent } from '../engine/rules'
import { loadManifest, loadModel, pickBest, sleep, think, V, type Manifest } from '../game'
import { Board, type Thought } from './Board'

const N = ringSize(V)
const HUMAN = 0
const AI = 1

const LEVELS = [
  { sims: 1, label: 'Chỉ trực giác (không tính trước)' },
  { sims: 50, label: 'Tính 50 nước giả định' },
  { sims: 200, label: 'Tính 200 nước giả định' },
  { sims: 800, label: 'Tính 800 nước giả định' },
  { sims: 3200, label: 'Tính 3.200 nước giả định (thắng minimax 8 nước)' },
]

function describe(a: number, player: number) {
  const sq = Math.floor(a / 2) + 1
  // người 0 ở dưới: +1 là sang phải; AI ở trên: +1 là sang trái trên màn hình
  const right = player === 0 ? a % 2 === 0 : a % 2 === 1
  return `ô ${sq} ${right ? '→' : '←'}`
}

const END_REASON: Record<string, string> = {
  'no-quan': 'Hết quan, tàn dân, thu quân kéo về.',
  'no-stones': 'Hết dân để rải, ván dừng.',
  'max-ply': 'Quá 300 nước, ván dừng.',
}

export function Play() {
  const [manifest, setManifest] = useState<Manifest | null>(null)
  const [genIdx, setGenIdx] = useState(0)
  const [level, setLevel] = useState(4)
  const [net, setNet] = useState<Net | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [st, setSt] = useState<Int32Array>(() => newGame(V))
  const [display, setDisplay] = useState<Int32Array>(() => newGame(V))
  const [hand, setHand] = useState<{ pos: number; count: number } | null>(null)
  const [flash, setFlash] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [thought, setThought] = useState<Thought | null>(null)
  const [msg, setMsg] = useState('Chọn một ô có dân ở hàng của bạn, rồi chọn chiều rải.')
  const [history, setHistory] = useState<string[]>([])
  const [lastAction, setLastAction] = useState<{ player: number; a: number } | null>(null)
  const [predicted, setPredicted] = useState<number | null>(null)
  const [aiFirst, setAiFirst] = useState(false)
  const [selected, setSelected] = useState<number | null>(null)
  const gameId = useRef(0)

  useEffect(() => {
    loadManifest()
      .then((m) => {
        setManifest(m)
        setGenIdx(m.models.length - 1)
      })
      .catch((e) => setError(String(e.message ?? e)))
  }, [])

  useEffect(() => {
    if (!manifest) return
    loadModel(manifest, manifest.models[genIdx])
      .then(setNet)
      .catch((e) => setError(String(e.message ?? e)))
  }, [manifest, genIdx])

  // dự đoán chênh lệch cuối ván theo góc nhìn người chơi = tỉ số hiện tại + mạng đoán phần còn lại
  const updatePrediction = useCallback(
    (s: Int32Array, valueForMover?: number) => {
      if (!net) return
      const diff = scoreOf(s, V, HUMAN) - scoreOf(s, V, AI)
      if (isDone(s, V)) {
        setPredicted(diff)
        return
      }
      const v = valueForMover ?? evaluate(net, canonical(s, V), V).value * totalPoints(V)
      setPredicted(diff + (toMove(s, V) === HUMAN ? v : -v))
    },
    [net],
  )

  const animate = useCallback(async (before: Int32Array, events: SowEvent[], id: number) => {
    const d = before.slice()
    const mover = before[N + 4]
    let handPos = -1
    let handCount = 0
    const drops = events.filter((e) => e.kind === 'drop').length
    const delay = drops > 40 ? 45 : drops > 20 ? 70 : 105
    for (const e of events) {
      if (gameId.current !== id) return false
      if (e.kind === 'spread') {
        const base = 1 + e.player * (V.squares + 1)
        for (let i = 0; i < V.squares; i++) d[base + i] = 1
        d[N + 2 + e.player] -= V.squares
        setDisplay(d.slice())
        await sleep(350)
      } else if (e.kind === 'pick') {
        handPos = e.pos
        handCount = e.count
        d[e.pos] = 0
        setDisplay(d.slice())
        setHand({ pos: handPos, count: handCount })
        await sleep(delay * 1.6)
      } else if (e.kind === 'drop') {
        handPos = e.pos
        handCount -= 1
        d[e.pos] += 1
        setDisplay(d.slice())
        setHand({ pos: handPos, count: handCount })
        await sleep(delay)
      } else if (e.kind === 'capture') {
        setHand(null)
        setFlash(e.pos)
        await sleep(260)
        d[e.pos] = 0
        if (e.quan) d[N + (e.pos === 0 ? 0 : 1)] = 0
        d[N + 2 + mover] += e.dan + (e.quan ? V.quan : 0)
        setDisplay(d.slice())
        await sleep(240)
        setFlash(null)
      }
    }
    setHand(null)
    return true
  }, [])

  const applyMove = useCallback(
    async (cur: Int32Array, a: number, id: number) => {
      const player = toMove(cur, V)
      const next = cur.slice()
      const events: SowEvent[] = []
      step(next, a, V, events)
      setLastAction({ player, a })
      setHistory((h) => [...h, `${player === HUMAN ? 'Bạn' : 'AI'} ${describe(a, player)}`])
      const ok = await animate(cur, events, id)
      if (!ok) return null
      setSt(next)
      setDisplay(next)
      const end = events.find((e) => e.kind === 'end')
      return { next, end: end && end.kind === 'end' ? end.reason : null }
    },
    [animate],
  )

  const finish = useCallback((s: Int32Array, reason: string | null) => {
    const me = scoreOf(s, V, HUMAN)
    const ai = scoreOf(s, V, AI)
    const verdict = me > ai ? `Bạn thắng ${me}–${ai}.` : me < ai ? `AI thắng ${ai}–${me}.` : `Hoà ${me}–${ai}.`
    setMsg(`${reason ? END_REASON[reason] + ' ' : ''}${verdict}`)
  }, [])

  const aiTurn = useCallback(
    async (cur: Int32Array, id: number) => {
      if (!net) return
      let s = cur
      while (!isDone(s, V) && toMove(s, V) === AI) {
        setMsg('AI đang tính…')
        await sleep(30)
        const r = think(net, s, LEVELS[level].sims)
        if (gameId.current !== id) return
        setThought({ visits: r.visits, q: r.q })
        const a = pickBest(r)
        updatePrediction(s, r.q[a])
        await sleep(LEVELS[level].sims > 1 ? 900 : 400)
        setThought(null)
        setMsg(`AI đi ${describe(a, AI)}.`)
        const res = await applyMove(s, a, id)
        if (!res) return
        s = res.next
        if (res.end) {
          finish(s, res.end)
          updatePrediction(s)
          return
        }
      }
      updatePrediction(s)
      setMsg(
        legalMask(s, V).every(Boolean) && Array.from({ length: V.squares }, (_, i) => s[1 + i]).every((x) => x === 0)
          ? 'Hàng bạn hết dân: chọn ô bất kỳ, 5 dân sẽ được rải từ kho của bạn trước.'
          : 'Đến lượt bạn.',
      )
    },
    [net, level, applyMove, finish, updatePrediction],
  )

  const onHumanMove = useCallback(
    async (a: number) => {
      if (busy || isDone(st, V) || toMove(st, V) !== HUMAN) return
      setBusy(true)
      const id = gameId.current
      const res = await applyMove(st, a, id)
      if (res) {
        if (res.end) {
          finish(res.next, res.end)
          updatePrediction(res.next)
        } else {
          await aiTurn(res.next, id)
        }
      }
      if (gameId.current === id) setBusy(false)
    },
    [busy, st, applyMove, aiTurn, finish, updatePrediction],
  )

  const newRound = useCallback(
    async (aiStarts: boolean) => {
      gameId.current += 1
      const id = gameId.current
      const s = newGame(V)
      if (aiStarts) s[N + 4] = AI
      setSt(s)
      setDisplay(s)
      setHand(null)
      setFlash(null)
      setThought(null)
      setHistory([])
      setLastAction(null)
      setAiFirst(aiStarts)
      setMsg('Chọn một ô có dân ở hàng của bạn, rồi chọn chiều rải.')
      updatePrediction(s)
      if (aiStarts && net) {
        setBusy(true)
        await aiTurn(s, id)
        if (gameId.current === id) setBusy(false)
      } else {
        setBusy(false)
      }
    },
    [net, aiTurn, updatePrediction],
  )

  useEffect(() => {
    if (net && plyOf(st, V) === 0 && !busy) updatePrediction(st)
    // chỉ cần chạy lại khi đổi model
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [net])

  const humanTurn = !busy && !isDone(st, V) && toMove(st, V) === HUMAN && net !== null
  const mask = humanTurn ? legalMask(st, V) : new Array(2 * V.squares).fill(false)
  const model = manifest?.models[genIdx]
  const pct = predicted === null ? 50 : Math.max(3, Math.min(97, 50 + (predicted / 40) * 50))

  return (
    <section className="arena" id="choi">
      <div className="board-shell">
        <Board
          v={V}
          st={display}
          hand={hand}
          flash={flash}
          playable={mask}
          onMove={onHumanMove}
          thought={thought}
          lastAction={lastAction}
          selected={selected}
          onSelect={setSelected}
        />
        {selected !== null && mask[2 * selected] && (
          <div className="dir-bar">
            <span>Ô {selected + 1}:</span>
            <button
              className="btn solid"
              onClick={() => {
                setSelected(null)
                onHumanMove(2 * selected + 1)
              }}
            >
              ← Rải sang trái
            </button>
            <button
              className="btn solid"
              onClick={() => {
                setSelected(null)
                onHumanMove(2 * selected)
              }}
            >
              Rải sang phải →
            </button>
          </div>
        )}
        <div className="status" aria-live="polite">
          <span className="msg">{error ? `Lỗi: ${error}` : net ? msg : 'Đang tải model…'}</span>
        </div>
      </div>

      <aside className="panel">
        <div className="scores">
          <div className="score you">
            <span>Bạn</span>
            <b>{scoreOf(display, V, HUMAN)}</b>
          </div>
          <div className="score ai">
            <span>AI</span>
            <b>{scoreOf(display, V, AI)}</b>
          </div>
        </div>

        <div>
          <h3>AI đoán kết quả</h3>
          <div className="evalbar" role="img" aria-label="Thanh dự đoán">
            <i style={{ width: `${pct}%` }} />
          </div>
          <div className="eval-text">
            {predicted === null
              ? '…'
              : Math.abs(predicted) < 0.5
                ? 'Ngang ngửa.'
                : predicted > 0
                  ? `Bạn hơn khoảng ${predicted.toFixed(0)} điểm khi hết ván.`
                  : `AI hơn khoảng ${(-predicted).toFixed(0)} điểm khi hết ván.`}
          </div>
        </div>

        <label className="field">
          <span>Đối thủ: AI sau bao nhiêu thế hệ tự học</span>
          <select value={genIdx} onChange={(e) => setGenIdx(Number(e.target.value))} disabled={busy || !manifest}>
            {manifest?.models.map((m, i) => (
              <option key={m.file} value={i}>
                Thế hệ {m.gen}
                {m.elo !== null ? ` · Elo ${Math.round(m.elo)}` : ''}
                {i === manifest.models.length - 1 ? ' (mạnh nhất)' : ''}
              </option>
            ))}
          </select>
          {model && model.gen === 0 && <span className="hint">Thế hệ 0 là mạng chưa học gì, gần như đi bừa.</span>}
        </label>

        <label className="field">
          <span>Mức suy nghĩ mỗi nước</span>
          <select value={level} onChange={(e) => setLevel(Number(e.target.value))} disabled={busy}>
            {LEVELS.map((l, i) => (
              <option key={l.sims} value={i}>
                {l.label}
              </option>
            ))}
          </select>
        </label>

        <div className="btn-row">
          <button className="btn solid" onClick={() => newRound(false)} disabled={!net}>
            Ván mới, bạn đi trước
          </button>
          <button className="btn" onClick={() => newRound(true)} disabled={!net}>
            AI đi trước
          </button>
        </div>

        {history.length > 0 && (
          <div>
            <h3>Các nước đã đi{aiFirst ? ' (AI đi trước)' : ''}</h3>
            <ol className="movelist">
              {history.map((h, i) => (
                <li key={i}>
                  {i + 1}. {h}
                </li>
              ))}
            </ol>
          </div>
        )}
      </aside>
    </section>
  )
}
