# oanquan-zero

[![ci](https://github.com/DuongCodeAI/oanquan-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/DuongCodeAI/oanquan-zero/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![python](https://img.shields.io/badge/python-3.10%2B-3776AB)

<p align="left">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"> <img src="https://img.shields.io/badge/AlphaZero%20t%E1%BB%B1%20vi%E1%BA%BFt-8E44AD?style=for-the-badge" alt="AlphaZero tự viết"> <img src="https://img.shields.io/badge/numba-00A3E0?style=for-the-badge" alt="numba"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"> <img src="https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript"> <img src="https://img.shields.io/badge/CPU%20only-2ECC71?style=for-the-badge" alt="CPU only"> <a href="https://oanquan-zero.vercel.app"><img src="https://img.shields.io/badge/Live%20demo-000000?style=for-the-badge&logo=vercel&logoColor=white" alt="Live demo"></a>
</p>

AI tự học **Ô ăn quan** từ số 0 theo cách AlphaZero: chỉ biết luật, tự đánh với chính nó, không có ván nào của
người. Luật chơi, solver, MCTS, vòng train và bản web đều tự viết. Toàn bộ chạy trên **laptop không GPU**
(Ryzen 5 5625U, 6 nhân, 16 GB RAM).

**Demo:** https://oanquan-zero.vercel.app (chơi với AI ngay trên trình duyệt, không có server)

> **English summary.** AlphaZero-style self-play for Ô ăn quan, a Vietnamese mancala game, written from scratch
> (rules, MCTS with per-edge rewards à la MuZero, training loop) and trained on a laptop CPU. The interesting part is
> evaluation against ground truth: I shrank the board until an exact solver could solve every reachable position
> (82,885 positions for 3 pits × 2 stones), then measured how often the agent picks a provably optimal move on a fixed
> test set: 37% → 85% for the raw policy, 77% → 92% with 160-simulation MCTS. On the full board there is no ground
> truth, so I use pair-based Elo against minimax baselines: at 200 simulations the agent beats greedy and 4-ply minimax
> but loses to 8-ply minimax, which examines ~30x more positions per move; with 3,200 simulations (about half of
> minimax's positions) the same network beats it 0.665 ± 0.047 over 100 game pairs. Two wrong conclusions I made and
> corrected along the way are documented below. The web demo runs a TypeScript port that is tested move-for-move against Python.

## Kết quả chính

**Bàn thu nhỏ 3 ô × 2 dân (có đáp án chính xác).** Solver duyệt hết 82.885 thế cờ. Train AI giống hệt bàn chuẩn
(16.000 ván tự đánh, 7,6 phút), rồi chấm trên **một bộ đề cố định 1.033 thế** lấy từ ván của người chơi hoàn hảo
(chỉ tính thế có nước tồi để chọn nhầm). Số từ `results/eval_small.json`.

| | trước train (thế hệ 0) | sau 40 thế hệ |
|---|---|---|
| chọn đúng nước tối ưu, chỉ dùng mạng (không tìm kiếm) | 37,5% | **85,0%** |
| chọn đúng nước tối ưu, MCTS 160 lượt | 77,0% | **91,7%** |
| sai số mạng đoán điểm còn lại | 4,09 điểm | **1,90 điểm** |
| điểm mất mỗi ván khi đấu người chơi hoàn hảo (200 ván) | 2,35 | **0,17** |

**Bàn chuẩn 5 ô × 5 dân (không có đáp án).** Đo bằng đấu theo **cặp ván**: hai ván cùng khai cuộc, đổi bên, cộng
chênh lệch điểm (lý do ở mục "Lần train 2").

Elo (MLE Bradley-Terry trên kết quả cặp ván, mốc 0 = đi ngẫu nhiên, 20 cặp mỗi trận, AI dùng MCTS 200 lượt;
`results/elo.json`):

| người chơi | Elo | ghi chú |
|---|---:|---|
| đi ngẫu nhiên | 0 | |
| tham ăn (ăn nhiều nhất ngay) | 497 | |
| AlphaZero thế hệ 0 (mạng chưa học, chỉ có MCTS) | 1.042 | |
| minimax 4 nước | 1.176 | 126 thế cờ / nước |
| AlphaZero thế hệ 10 | 1.324 | |
| AlphaZero thế hệ 40 | 1.376 | |
| **AlphaZero thế hệ 100** | **1.417** | 200 thế cờ / nước |
| minimax 8 nước | 1.517 | 5.911 thế cờ / nước |

Trận quyết định đo lại với 100 cặp (`results/pair_matches.jsonl`, điểm cặp ± sai số chuẩn):

| trận | cặp thắng - hoà - thua | điểm cặp |
|---|---|---|
| thế hệ 100 vs thế hệ 40 | 69 - 8 - 23 | **0,73 ± 0,04** |
| thế hệ 100 vs thế hệ 80 (cuối lần train 1) | 43 - 23 - 34 | 0,55 ± 0,05 |
| thế hệ 100 vs minimax 8 nước | 15 - 19 - 66 | 0,25 ± 0,04 |

Trước và sau train, cùng MCTS 200 lượt, chỉ khác trọng số (100 cặp mỗi trận, điểm cặp):

| đối thủ | thế hệ 0 (chưa học) | thế hệ 100 |
|---|---|---|
| tham ăn | 0,94 | **0,985** (97 - 3 - 0) |
| minimax 4 nước | 0,31 | **0,765** (71 - 11 - 18) |
| minimax 8 nước | 0,055 | **0,245** (15 - 19 - 66) |

Với 200 lượt MCTS, AI thắng đậm đi ngẫu nhiên, tham ăn và minimax 4 nước, nhưng thua minimax 8 nước, đối thủ xét
nhiều thế cờ hơn nó ~30 lần mỗi nước. **Cho cùng mạng đó tìm kiếm nhiều hơn thì thắng** (100 cặp mỗi mức):

| thế hệ 100 vs minimax 8 nước | thế cờ xét / nước | cặp thắng - hoà - thua | điểm cặp |
|---|---|---|---|
| MCTS 200 lượt | 200 (≈ 1/30 minimax 8) | 15 - 19 - 66 | 0,245 ± 0,043 |
| MCTS 800 lượt | 800 (≈ 1/7) | 34 - 29 - 37 | 0,485 ± 0,050 |
| **MCTS 3.200 lượt** | 3.200 (≈ 1/2) | **54 - 25 - 21** | **0,665 ± 0,047** |

3.200 lượt mất ~0,3 s mỗi nước trên trình duyệt; đó là mức mặc định của web demo.

## Cách làm

```
luật (numba) ──► MCTS + mạng policy/value ──► tự đánh N ván ──► (bàn, phân bố lượt thăm, điểm thật về sau)
     ▲                                                                   │
     └──────────── mạng mới ◄── train: CE(policy) + MSE(value), lật gương ◄┘     mỗi vòng = 1 thế hệ
```

- **Luật** viết bằng numba trên một vòng 12 ô. Xoay vòng 6 ô = đổi góc nhìn người đi, nên mạng chỉ học một góc nhìn.
  Đối xứng gương trái-phải (có test chứng minh) dùng để nhân đôi dữ liệu.
- **Giá trị thế cờ = chênh lệch điểm từ đây tới hết ván.** Ăn quân chỉ cộng cho người ăn, rải khi hết dân luôn tốn
  đúng 5 điểm, nên giá trị chỉ phụ thuộc bàn cờ, so trực tiếp được với solver. Đổi lại AI tối đa hoá cách biệt chứ
  không tối đa hoá xác suất thắng.
- **MCTS có phần thưởng trên cạnh** (như MuZero) vì Ô ăn quan ăn điểm ngay trong lúc đi: `Q(x,a) = r(x,a) − V(con)`,
  Q chuẩn hoá min-max trong cây. 200 lượt mô phỏng mất ~2 ms (numba, mạng 49k tham số).
- **Mạng**: MLP 3 lớp ẩn × 128, 49.035 tham số. Nhỏ vì phải chạy hàng trăm lượt mỗi nước trên CPU và trình duyệt.
- **Solver chính xác** cho bàn thu nhỏ: BFS hết thế cờ đến được, rồi value iteration (Bellman). Kiểm tra chéo với
  minimax alpha-beta (hai thuật toán độc lập, phải ra cùng số).
- **Web**: luật, mạng và MCTS viết lại bằng TypeScript. Test so với fixtures sinh từ Python: luật khớp trên 120 ván
  ngẫu nhiên, logits sai < 1e-4, số lượt thăm MCTS trùng hoàn toàn. Demo dùng đúng model đã đo, không phải bản na ná.

## Lần train 1 và lần train 2 (bàn chuẩn)

| | lần 1 | lần 2 (chạy tiếp từ lần 1) |
|---|---|---|
| thế hệ | 1 → 80 | 81 → 100 |
| ván tự đánh | 64.000 (800 / thế hệ) | 20.000 (1.000 / thế hệ) |
| lượt MCTS mỗi nước | 200 | 400 |
| learning rate | 1e-3 | 5e-4 |
| thế cờ sinh ra để train | 1.779.613 | 580.265 |
| thời gian (laptop, 6 tiến trình) | 25,2 phút | 11,6 phút |
| loss policy (đầu → cuối) | 1,58 → 0,84 | 0,77 → 0,70 |
| loss value (đầu → cuối) | 0,040 → 0,017 | 0,014 → 0,012 |

**Lần 2 không làm AI mạnh hơn một cách đo được.** Kết quả đo 100 cặp ván ở bảng trên: thế hệ 100 vs thế hệ 80 là
0,55 ± 0,05 (chưa phân biệt được), đấu minimax 8 nước thì như thế hệ 40 (~0,25). Loss vẫn giảm nhẹ (policy 0,77 →
0,70) nhưng sức chơi không tăng theo.

Trên đường đi mình kết luận sai hai lần, ghi lại vì đó là phần học được nhiều nhất:
- **Thước đo hỏng.** Người đi trước thắng 77-90% số ván ở mọi trận, nên tính thắng/thua từng ván thì trận nào cũng
  ra ~50%. Minimax 8 nước đấu minimax 2 nước: tính từng ván thắng 69%, tính theo cặp ván thắng 40/40. Mình đã chạy
  lần 2 trước khi nhận ra. Giờ mọi số trên bàn chuẩn đều tính theo cặp.
- **Mẫu quá nhỏ.** 40 cặp cho thấy thế hệ 90 không hơn thế hệ 40 (15-6-19), mình kết luận "chững lại từ thế hệ 40".
  Đo lại 100 cặp: thế hệ 100 thắng thế hệ 40 với 0,73 ± 0,04. Kết luận đúng là AI vẫn tiến bộ khi đấu phiên bản cũ
  của chính nó, nhưng không tiến bộ khi đấu minimax 8 nước (self-play không bắc cầu).

**Vì sao chưa thắng được minimax 8 nước** (cái đã đo và cái mới là giả thuyết):

1. *Đã đo:* **minimax 8 nước suy nghĩ nhiều hơn ~30 lần.** Số thế cờ được xét mỗi nước đi: minimax 4 nước 126,
   6 nước 1.060, **8 nước 5.911**, 10 nước 30.317; AlphaZero 200 lượt xét **200**. Ván chỉ dài ~29 nước, nên nhìn
   chính xác 8 nước là đã thấy gần 1/3 ván. Cho AI 800 lượt thì cách biệt với minimax 8 đổi từ −5,4 điểm/ván
   (50 lượt) thành +1,8 điểm/ván (40 ván): tìm kiếm thêm có giúp.
2. *Đã đo:* **self-play không bắc cầu.** Thế hệ 100 thắng thế hệ 40 rõ ràng nhưng đấu minimax 8 thì ngang thế hệ 40.
   Dữ liệu tự đánh chỉ dạy AI đối phó với chính nó.
3. *Đã đo (A/B):* **mạng to hơn không phải lời giải.** Thí nghiệm "v2": mạng 164k tham số (gấp 3), target value =
   trung bình kết quả thật và đánh giá MCTS ở gốc, chỉ 6 nước đầu chọn ngẫu nhiên (thay vì 12, gần nửa ván 29 nước).
   Cùng 30 thế hệ × 800 ván × 200 lượt như lần 1 (17 phút). Kết quả 100 cặp mỗi trận:
   | trận | điểm cặp |
   |---|---|
   | v2 thế hệ 30 vs lần 1 thế hệ 30 | **0,605 ± 0,049** |
   | v2 thế hệ 30 vs lần 1 thế hệ 100 | 0,575 ± 0,049 |
   | v2 thế hệ 30 vs minimax 8 nước | 0,325 ± 0,047 |
   | lần 1 thế hệ 30 vs minimax 8 nước | 0,345 ± 0,048 |
   v2 học nhanh hơn khi so với chính dòng AI này, nhưng trước minimax 8 nước thì không khá hơn. Cộng với bảng "tìm
   kiếm nhiều hơn" ở trên: nút thắt là **độ sâu tìm kiếm**, không phải mạng. Web vẫn dùng mạng lần 1 (nhỏ hơn, đủ dùng).

Chi tiết từng bước, kể cả những lần đoán sai: [NOTES.md](NOTES.md).

## Solver: bàn thu nhỏ

Quan luôn bằng 2 ô dân như bàn chuẩn. Giá trị = người đi trước hơn bao nhiêu điểm khi cả hai chơi hoàn hảo.

| biến thể | thế cờ | giá trị thế đầu | thế không xác định |
|---|---:|---:|---:|
| 2 ô × 1 dân | 95 | −2 | 0 |
| 2 × 2 | 1.039 | +4 | 0 |
| 2 × 3 | 7.143 | +7 | 0 |
| 2 × 4 | 15.972 | −4 | 0 |
| 3 × 1 | 4.311 | +10 / 10 | 0 |
| 3 × 2 | 82.885 | −4 | 78 |
| 3 × 3 | 585.052 | +4 | 386 |
| 4 × 1 | 96.021 | −5 | 0 |

"Không xác định": thế nằm trên (hoặc dẫn vào) vòng lặp không ăn quân, giá trị dao động mãi; bị loại khỏi phép đo.
4 × 2 (~21 triệu thế) hết RAM laptop với cách lưu hiện tại (dict numba), chưa giải.

## Chạy lại

```bash
pip install -e ".[train,dev]"
pytest -q                                              # luật, đối xứng, solver vs alpha-beta
python scripts/solve_variants.py S3P2Q4                # giải bàn 3 × 2
python -m oanquan.train --variant S3P2Q4 --gens 40 --games 400 --sims 160 --steps 300 \
    --solution results/solved/S3P2Q4.npz --out runs/S3P2Q4
python scripts/eval_small.py runs/S3P2Q4 --gens 0 5 10 20 40
python -m oanquan.train --variant S5P5Q10 --gens 80 --games 800 --sims 200 --steps 800 --keep 8 --out runs/standard
python scripts/elo.py runs/standard --gens 0 10 40 100 --baselines random greedy minimax4 minimax8 --games 40
python scripts/pair_match.py runs/standard gen100 minimax8 --pairs 100       # PYTHONPATH="src;scripts" trên Windows
cd web && npm ci && npx vitest run && npm run dev      # web demo
```

## Luật dùng

5 ô dân × 5 dân mỗi bên, 2 quan (10 điểm); ô kế có dân thì bốc rải tiếp, gặp ô quan còn quân thì mất lượt; trống
rồi có quân thì ăn, ăn liên tiếp được; hết dân thì rải 5 dân từ kho (thiếu thì vay); hết cả hai ô quan thì thu quân.
Không dùng luật "quan non". Hai chỗ luật gốc không nói (tự định, ghi trong NOTES): kho hai bên không đủ dân để rải
thì ván dừng; ván lặp mãi thì phần trên bàn không tính, self-play cắt ở 300 nước.

## Cấu trúc

```
src/oanquan/   rules.py · solver.py · net.py · mcts.py · selfplay.py · train.py · players.py
scripts/       solve_variants.py · eval_small.py · elo.py · export_web.py · export_fixtures.py
tests/         test_rules.py · test_solver.py
web/           React + Vite; src/engine/*.ts là bản TypeScript của luật, mạng, MCTS (+ parity.test.ts)
results/       eval_small.json · elo.json · solved/summary.json
```
