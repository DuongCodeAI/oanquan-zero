# Ghi chú quyết định và những chỗ vấp

Ghi theo thứ tự làm, kể cả chỗ sai. Số liệu đều là số đã chạy thật trên laptop (Ryzen 5 5625U, không GPU).

## Luật chọn dùng
Ô ăn quan có nhiều biến thể theo vùng. Bản dùng ở đây:
- 5 ô dân mỗi bên, mỗi ô 5 dân; 2 ô quan, mỗi ô 1 quan = 10 điểm.
- Bốc một ô bên mình, chọn chiều, rải mỗi ô 1 dân (kể cả ô quan).
- Ô kế tiếp có dân thì bốc rải tiếp. Ô kế tiếp là ô quan còn quân thì mất lượt.
- Ô kế tiếp trống mà ô sau đó có quân thì ăn, rồi xét tiếp: trống - có quân thì ăn tiếp (ăn liên tiếp).
  Hai ô trống liền nhau thì mất lượt. Ô quan đã bị ăn sạch được coi là ô trống.
- Đến lượt mà hàng mình hết dân thì lấy S dân đã ăn ra rải, thiếu thì vay đối thủ, cuối ván trả.
- Hết cả hai ô quan (không còn quan, không còn dân trong ô quan) thì kết thúc, mỗi bên thu dân hàng mình.
- **Không** dùng luật "quan non" (cấm ăn quan khi ô quan có ít dân) vì mỗi vùng chơi một kiểu.

Hai chỗ luật gốc không nói, phải tự định (xem các mục bên dưới vì sao):
- Hết dân mà kho của cả hai bên cộng lại cũng không đủ S dân để rải: ván dừng, mỗi bên thu hàng mình.
- Ván lặp mãi không kết thúc: phần quân còn trên bàn không tính cho ai. Self-play cắt ở 300 nước.

## Biểu diễn
- Bàn là một vòng 12 ô: 0 = quan trái, 1..5 = hàng người 0, 6 = quan phải, 7..11 = hàng người 1.
  Xoay vòng 6 ô là đổi góc nhìn sang người kia, nên mạng chỉ cần học "góc nhìn người đang đi".
- Đối xứng gương: ô j → (6 − j) mod 12, đổi chiều rải. Có test kiểm tra đi nước lật trên bàn lật ra đúng
  bàn lật của kết quả gốc (đúng với mọi thế cờ ngẫu nhiên đã thử), nên dùng được để nhân đôi dữ liệu train.
- Viết luật bằng numba: ~21k nước/s kể cả overhead Python khi đánh ngẫu nhiên.

## Giá trị của thế cờ = chênh lệch điểm từ đây trở đi, không phải thắng/thua
- Ăn quân chỉ cộng cho người ăn; rải khi hết dân luôn tốn đúng S điểm (thiếu thì vay, cuối ván trả).
  Vì vậy "người đi kiếm thêm được bao nhiêu điểm hơn đối thủ từ giờ đến hết ván" chỉ phụ thuộc bàn cờ,
  không phụ thuộc tỉ số hiện tại. Mạng không cần biết tỉ số, và solver chỉ cần lưu bàn cờ.
- Đổi lại, AI tối đa hoá cách biệt chứ không tối đa hoá xác suất thắng. Khi đang dẫn xa hai thứ này gần
  như trùng nhau; khi sát nút thì có thể khác. Chấp nhận để so được trực tiếp với lời giải chính xác.

## Lỗi 1: vay dân "đẻ ra" dân mới, value iteration không hội tụ
- Lúc đầu mô hình vay bằng cách cho điểm âm thoải mái. Thống kê 20.000 ván ngẫu nhiên đã có dấu hiệu lạ:
  biến thể 2 ô × 3 dân rải trung bình **47 lần/ván** và **10%** số ván bị cắt ở 300 nước.
- Solver cho biến thể nhỏ nhất (95 thế cờ) chạy 7.000 vòng vẫn còn đúng 1.838 thế đổi giá trị mỗi vòng.
  Số thế cờ lớn hơn số thế "hợp lý" là đã sai: trên bàn có lúc nhiều dân hơn tổng số dân của ván.
- Nguyên nhân: điểm âm không giới hạn nghĩa là dân lấy từ hư không. Ngoài đời tổng số dân cố định, kho hai
  bên cộng lại không đủ thì không rải được.
- Sửa: thêm luật dừng ở trên, thêm test "trên bàn không bao giờ quá tổng số dân". Sau khi sửa, cùng biến thể
  hội tụ sau 4 vòng.

## Lỗi 2 (không hẳn là lỗi): có thế cờ không có giá trị xác định
- Biến thể 3 ô × 2 dân: 82.885 thế, sau 400 vòng value iteration vẫn còn ~30 thế đổi giá trị mỗi vòng, chênh
  ±1 điểm. Giá trị bị chặn (|v| ≤ 20) nên không phải lỗi "đẻ dân" như trên.
- Cách xử lý: chạy 3.000 vòng, thế nào còn đổi trong 1.000 vòng cuối thì đánh dấu "chưa xác định" và **loại khỏi
  mọi phép đo**, kể cả thế có con là thế chưa xác định (vì khi đó không biết nước nào tối ưu).
  3 × 2: 78/82.885 thế (0,09%), 3 × 3: 386/585.052 thế.
- Mình đoán sai nguyên nhân **hai lần** trước khi kiểm tra thật:
  1. "Chỉ xảy ra sau khi ăn hết cả hai quan to (ô quan còn dân nên ván chưa hết)". Đếm lại: chỉ 10% thế.
  2. "Ván ăn qua ăn lại rồi rải lại mãi". Đi theo nước tốt nhất từ 78 thế này thì không gặp vòng nào cả.
- Nguyên nhân thật: lấy đồ thị con chỉ gồm 78 thế này, tách thành phần liên thông mạnh. Ra **8 vòng, mỗi vòng
  đúng 8 nước, mọi nước trong vòng đều không ăn quân** (64 thế), cộng 14 thế không nằm trên vòng nhưng nước tốt
  nhất dẫn vào một vòng (vài nước trong số đó là hết dân phải rải, −3 điểm). Giá trị mỗi thế dao động 1 ↔ 2 với chu
  kỳ 4 vòng lặp: ván bị cắt ở nước thứ t, và ai "kẹt" lượt phải rời vòng phụ thuộc t.
- Lần thứ ba suýt sai: lúc đầu mình chỉ tìm được một vòng rồi viết luôn "các thế này nằm trên vòng không ăn quân".
  Viết test kiểm tra cho cả 78 thế thì test đỏ, vì 14 thế dẫn-vào-vòng có nước −3. Câu đúng là câu ở trên.
- Ai rời vòng cũng chỉ được 1-2 điểm, nên đây là lựa chọn rất nhỏ; loại khỏi phép đo không ảnh hưởng kết luận.

## Solver: số thế cờ đến được từ thế khởi đầu
| biến thể | thế cờ | value iteration | chưa xác định | giá trị thế đầu (người đi trước) |
|---|---:|---|---:|---:|
| 2 ô × 1 dân | 95 | 4 vòng | 0 | −2 |
| 2 × 2 | 1.039 | 7 | 0 | +4 |
| 2 × 3 | 7.143 | 12 | 0 | +7 |
| 2 × 4 | 15.972 | 18 | 0 | −4 |
| 3 × 1 | 4.311 | 16 | 0 | +10 (thắng trắng) |
| 3 × 2 | 82.885 | 3.000 (không hội tụ hết) | 78 | −4 |
| 3 × 3 | 585.052 | 3.000 | 386 | +4 |
| 4 × 1 | 96.021 | 27 | 0 | −5 |

Quan luôn = 2 ô dân (như bàn chuẩn: 5 dân, quan 10). Lợi thế đi trước đổi dấu liên tục theo kích thước bàn,
không có quy luật đơn giản kiểu "đi trước luôn lợi".

## MCTS và mạng
- Ô ăn quan có điểm ngay trong lúc đi, nên MCTS cần phần thưởng trên cạnh như MuZero:
  Q(x, a) = r(x, a) − V(con). Q chuẩn hoá min-max trong cây vì độ lớn Q đổi nhiều theo giai đoạn ván.
- Nước chưa thăm lấy Q = giá trị mạng ước lượng của chính nút cha (thay vì 0).
- Mạng: MLP 3 lớp ẩn 128, ~50k tham số. Mỗi ô mã hoá one-hot số dân 0..6, ≥7 gộp, cộng số dân chia 10.
- MCTS viết bằng numba: 160 lượt mô phỏng mất 2,2 ms trên laptop. Self-play 400 ván 3 × 2 mất ~7 s với 6 tiến
  trình, nên không cần GPU, không cần Colab.
- Forward numba khớp PyTorch tới 1e-8. Bản TypeScript trên web khớp bản Python: luật trên 120 ván ngẫu nhiên (3
  biến thể), logits sai < 1e-4, số lượt thăm MCTS 64 lượt trùng hoàn toàn trên 40 thế thử.
- Lỗi vặt: numba từ chối `x @ W` vì `np.maximum(…, 0.0)` đẩy float32 lên float64; phải dùng `np.float32(0.0)`.
