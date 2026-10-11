import labJson from '../data/lab.json'
import { BeforeAfter, LineChart } from './Charts'

interface TrainSummary {
  variant: string
  gens: number
  games: number
  positions: number
  sims: number
  minutes: number
  games_per_gen: number
  steps: number
  batch: number
  hidden: number
  phases: { gens: number; games: number; sims: number }[]
}

interface EvalRow {
  gen: number
  policy_opt: number
  mcts_opt: number
  value_mae: number
  n_positions: number
  vs_perfect_margin: number
  vs_perfect_score: number
}

interface LabData {
  params: number
  train_small: TrainSummary
  train_standard: TrainSummary
  eval_small: { variant: string; sims: number; positions: number; rows: EvalRow[] }
  learning: { squares: number; stones: number; states: number }
  elo: { points: { gen: number; elo: number }[]; baselines: { label: string; elo: number }[]; note: string }
  standard_vs: { opponent: string; before: number; after: number; pairs: number }[]
  notes: { title: string; body: string }[]
}

const lab = labJson as unknown as LabData
const pct = (y: number) => `${Math.round(y * 100)}%`
const num = (x: number) => x.toLocaleString('vi-VN')

export function Lab() {
  const L = lab.learning
  const ev = lab.eval_small
  const first = ev.rows[0]
  const last = ev.rows[ev.rows.length - 1]
  const ts = lab.train_small
  const tb = lab.train_standard

  const E = lab.elo
  const best = E.points[E.points.length - 1] ?? { gen: tb.gens, elo: 0 }
  const eloVals = [0, ...E.points.map((p) => p.elo), ...E.baselines.map((b) => b.elo)]
  const eloMin = Math.floor(Math.min(...eloVals) / 200) * 200
  const eloMax = Math.ceil(Math.max(...eloVals) / 200) * 200

  return (
    <section className="lab" id="thi-nghiem">
      <h2 className="lab-head">Cách nó được huấn luyện</h2>
      <p className="hero-sub">
        Không có dữ liệu ván cờ nào của người. Chỉ có luật chơi và một vòng lặp, chạy trên CPU laptop (Ryzen 5 5625U, 6
        nhân), không GPU.
      </p>

      <ol className="loop">
        <li>
          <b>Tự đánh</b>
          Mạng hiện tại tự chơi {num(tb.phases[0].games)} ván. Mỗi nước, cây tìm kiếm MCTS thử {tb.phases[0].sims} nhánh
          giả định rồi chọn nước được thăm nhiều nhất.
        </li>
        <li>
          <b>Ghi lại</b>
          Mỗi thế cờ lưu: nước nào MCTS thăm nhiều, và cuối ván người đi thật sự hơn/kém bao nhiêu điểm kể từ thế đó.
        </li>
        <li>
          <b>Học</b>
          Mạng học bắt chước MCTS (policy) và đoán điểm còn lại (value). Mỗi thế được lật gương trái-phải để có gấp đôi dữ
          liệu.
        </li>
        <li>
          <b>Lặp lại</b>
          Mạng giỏi hơn làm MCTS giỏi hơn, MCTS giỏi hơn tạo dữ liệu tốt hơn. Mỗi vòng là một thế hệ.
        </li>
      </ol>
      <p className="loop-back">↻ thế hệ 0 là mạng khởi tạo ngẫu nhiên, chưa biết gì</p>

      <div className="specs">
        <div className="spec">
          <b>{num(lab.params)}</b>
          <span>tham số (MLP 3 lớp ẩn × {tb.hidden})</span>
        </div>
        <div className="spec">
          <b>{num(tb.games)}</b>
          <span>ván tự đánh trên bàn chuẩn, {tb.gens} thế hệ, 2 lần train</span>
        </div>
        <div className="spec">
          <b>{num(tb.minutes)} phút</b>
          <span>tổng thời gian train bàn chuẩn</span>
        </div>
        <div className="spec">
          <b>~2 ms</b>
          <span>một lần MCTS 200 lượt (numba)</span>
        </div>
      </div>

      <div className="finding">
        <div>
          <div className="kicker">Chấm điểm bằng đáp án thật</div>
          <h3>Trước và sau khi train</h3>
          <p>
            Bàn chuẩn quá lớn để biết nước nào là tốt nhất. Nên mình thu nhỏ bàn còn {L.squares} ô × {L.stones} dân, viết
            solver duyệt hết <strong>{num(L.states)} thế cờ</strong> và tính nước tối ưu của từng thế, rồi train AI trên
            bàn đó đúng như bàn chuẩn ({num(ts.games)} ván, {num(ts.minutes)} phút).
          </p>
          <p>
            Đề thi cố định: {num(first.n_positions)} thế cờ lấy từ ván của người chơi hoàn hảo, chỉ tính thế có nước tồi để
            chọn nhầm. Thế hệ nào cũng làm cùng bộ đề này.
          </p>
          <p>
            Sau {last.gen} thế hệ, sai số khi đoán trước kết quả giảm từ {num(+first.value_mae.toFixed(1))} xuống{' '}
            <strong>{num(+last.value_mae.toFixed(1))} điểm</strong>. Đấu với người chơi hoàn hảo, mỗi ván mất trung bình{' '}
            {num(+(-first.vs_perfect_margin).toFixed(2))} điểm lúc chưa học, còn{' '}
            <strong>{num(+(-last.vs_perfect_margin).toFixed(2))}</strong> sau khi học.
          </p>
        </div>
        <BeforeAfter
          beforeLabel="Trước train"
          afterLabel={`Thế hệ ${last.gen}`}
          rows={[
            { label: 'Chọn đúng nước tối ưu, chỉ bằng trực giác', before: first.policy_opt, after: last.policy_opt, format: pct, max: 1 },
            { label: `Chọn đúng nước tối ưu, có MCTS ${ev.sims} lượt`, before: first.mcts_opt, after: last.mcts_opt, format: pct, max: 1 },
            {
              label: 'Sai số đoán điểm còn lại (thấp hơn là tốt)',
              before: first.value_mae,
              after: last.value_mae,
              format: (x) => `${num(+x.toFixed(1))} đ`,
              max: Math.ceil(first.value_mae),
            },
            {
              label: 'Điểm mất mỗi ván so với chơi hoàn hảo (thấp hơn là tốt)',
              before: -first.vs_perfect_margin,
              after: -last.vs_perfect_margin,
              format: (x) => `${num(+x.toFixed(2))} đ`,
              max: Math.ceil(-first.vs_perfect_margin),
            },
          ]}
        />
      </div>

      <div className="finding">
        <div>
          <div className="kicker">Theo từng thế hệ</div>
          <h3>Học nhanh nhất ở mấy thế hệ đầu</h3>
          <p>
            Trực giác (chỉ nhìn bàn cờ rồi chọn, không tính trước) đi từ {pct(first.policy_opt)} lên {pct(last.policy_opt)}.
            Có MCTS thì cao hơn hẳn, kể cả ở thế hệ 0: tìm kiếm trên luật thật tự nó đã mạnh, mạng chỉ giúp tìm đúng chỗ.
          </p>
          <p>Khoảng cách giữa hai đường hẹp dần: mạng học được dần những gì trước đây phải tính mới thấy.</p>
        </div>
        <LineChart
          caption="Tỉ lệ chọn đúng nước tối ưu theo thế hệ, trên bộ đề cố định"
          series={[
            { key: 'm', label: 'có MCTS', color: '--s2', points: ev.rows.map((r) => ({ x: r.gen, y: r.mcts_opt })) },
            { key: 'p', label: 'chỉ trực giác', color: '--s1', points: ev.rows.map((r) => ({ x: r.gen, y: r.policy_opt })) },
          ]}
          yMin={0.3}
          yMax={1}
          yTicks={[0.4, 0.6, 0.8, 1]}
          yFormat={pct}
          xLabel="thế hệ"
        />
      </div>

      {E.points.length > 0 && (
        <div className="finding">
          <div>
            <div className="kicker">Bàn chuẩn 5 ô × 5 dân</div>
            <h3>
              Elo theo thế hệ: từ {Math.round(E.points[0].elo)} lên {Math.round(best.elo)}
            </h3>
            <p>
              Bàn chuẩn không có đáp án, nên đo bằng đấu vòng tròn: các thế hệ đấu nhau và đấu với đối thủ viết tay,{' '}
              <strong>tham ăn</strong> (luôn ăn nhiều nhất ngay) và <strong>minimax</strong> nhìn trước 4 và 8 nước. Mỗi
              trận tính theo cặp ván (cùng khai cuộc, đổi bên) vì người đi trước thắng tới 77-90% số ván. Elo ước lượng
              từ toàn bộ kết quả, mốc 0 là người đi ngẫu nhiên.
            </p>
            {E.note && <p>{E.note}</p>}
          </div>
          <LineChart
            caption="Elo theo thế hệ trên bàn chuẩn"
            series={[{ key: 'elo', label: 'AlphaZero', color: '--s1', points: E.points.map((p) => ({ x: p.gen, y: p.elo })) }]}
            refs={E.baselines.map((b) => ({ y: b.elo, label: b.label }))}
            yMin={eloMin}
            yMax={eloMax}
            yTicks={Array.from({ length: (eloMax - eloMin) / 200 + 1 }, (_, i) => eloMin + i * 200)}
            yFormat={(y) => String(Math.round(y))}
            xLabel="thế hệ"
          />
        </div>
      )}

      {lab.standard_vs.length > 0 && (
        <div className="finding">
          <div>
            <div className="kicker">Bàn chuẩn: trước và sau</div>
            <h3>Tỉ lệ thắng cặp ván trước các đối thủ viết tay</h3>
            <p>
              Cùng một AI với MCTS 200 lượt, chỉ khác bộ trọng số: thế hệ 0 (chưa học) và thế hệ cuối. Tỉ lệ tính theo
              cặp ván ({lab.standard_vs[0]?.pairs ?? 0} cặp mỗi đối thủ), hoà tính nửa.
            </p>
          </div>
          <BeforeAfter
            beforeLabel="Trước train"
            afterLabel={`Thế hệ ${best.gen}`}
            rows={lab.standard_vs.map((r) => ({ label: `với ${r.opponent}`, before: r.before, after: r.after, format: pct, max: 1 }))}
          />
        </div>
      )}

      <div className="finding">
        <div>
          <div className="kicker">Những chỗ vấp</div>
          <h3>Ghi lại cả những lần sai</h3>
          <p>
            Chi tiết từng lỗi, cách phát hiện và cách sửa nằm trong{' '}
            <a href="https://github.com/DuongCodeAI/oanquan-zero/blob/main/NOTES.md">NOTES.md</a>.
          </p>
        </div>
        <ul className="notes-list">
          {lab.notes.map((n) => (
            <li key={n.title}>
              <b>{n.title}</b>
              <span>{n.body}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
