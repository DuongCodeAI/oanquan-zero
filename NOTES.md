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
