import { Lab } from './components/Lab'
import { Play } from './components/Play'
import lab from './data/lab.json'

const num = (x: number) => x.toLocaleString('vi-VN')
const T = lab.train_standard

export default function App() {
  return (
    <div className="wrap">
      <header className="top">
        <div className="brand">
          Ô ăn quan Zero<small>AlphaZero · tự học từ số 0</small>
        </div>
        <nav>
          <a href="#choi">Chơi</a>
          <a href="#thi-nghiem">Thí nghiệm</a>
          <a href="https://github.com/DuongCodeAI/oanquan-zero">Mã nguồn</a>
        </nav>
      </header>

      <h1 className="hero-title">Một AI chưa từng xem ai chơi Ô ăn quan.</h1>
      <p className="hero-sub">
        Nó chỉ được biết luật. Mình cho nó tự đánh với chính nó <strong>{num(T.games)} ván</strong> qua{' '}
        <strong>{T.gens} thế hệ</strong> (
        {T.phases
          .map((p) => `${p.gens} thế hệ × ${num(p.games)} ván, MCTS ${p.sims} lượt mỗi nước`)
          .join(', rồi ')}
        ), tổng cộng <strong>{num(T.minutes)} phút</strong> trên laptop Ryzen 5 5625U (6 nhân, 16 GB RAM,{' '}
        <strong>không GPU</strong>). Bộ não của nó là một mạng {num(lab.params)} tham số. Mọi nước nó đi bên dưới được
        tính ngay trong trình duyệt của bạn; chấm đỏ là các nước nó đang cân nhắc.
      </p>

      <Play />
      <Lab />

      <footer>
        Tiến Dương · <a href="https://github.com/DuongCodeAI/oanquan-zero">github.com/DuongCodeAI/oanquan-zero</a> · luật,
        solver và MCTS viết từ đầu (Python + numba), web chạy cùng engine viết lại bằng TypeScript, kiểm tra khớp từng nước
        với bản Python.
      </footer>
    </div>
  )
}
