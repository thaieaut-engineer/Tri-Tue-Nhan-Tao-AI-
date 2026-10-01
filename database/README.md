# Hướng Dẫn Cơ Sở Dữ Liệu TourAI (`chatbot_tour`)

Thư mục này chứa toàn bộ định nghĩa cấu trúc bảng và dữ liệu mẫu của hệ thống TourAI.

---

## 1. Danh sách các tệp cơ sở dữ liệu

| Tên tệp | Kích thước | Mô tả |
| :--- | :---: | :--- |
| [`chatbot_tour_full.sql`](chatbot_tour_full.sql) | ~930 KB | **Bản dump CSDL hoàn chỉnh nhất**: Bao gồm toàn bộ cấu trúc 10 bảng chuẩn hóa, 6 danh mục, 20 tours, 52 lịch trình ngày, **1.290 câu hỏi đáp tri thức AI** và 3 tài khoản mặc định (`admin`, `kenny`, `user`). |
| [`schema.sql`](schema.sql) | ~12 KB | File định nghĩa cấu trúc khung DDL 10 bảng cơ bản kèm dữ liệu tour mẫu ban đầu. |
| [`db.py`](db.py) | ~1 KB | Module Python quản lý kết nối MySQL với `mysql.connector` và nạp cấu hình từ `.env`. |

---

## 2. Hướng dẫn Import vào DBeaver / CloudBeaver

### Cách 1: Sử dụng DBeaver Desktop hoặc CloudBeaver (Web)
1. Mở DBeaver / CloudBeaver, kết nối đến MySQL (`localhost:3306`).
2. Mở trình soạn thảo SQL (**SQL Editor**).
3. Mở tệp [`database/chatbot_tour_full.sql`](chatbot_tour_full.sql) (hoặc copy toàn bộ nội dung tệp dán vào SQL Editor).
4. Nhấn **Execute SQL Script** (phím tắt `Alt + X` hoặc biểu tượng nút Play có trang giấy).
5. Sau khi chạy xong, nhấn nút **Refresh** (`F5`) ở cây điều hướng bên trái: database `chatbot_tour` sẽ xuất hiện đầy đủ 10 bảng.

### Cách 2: Sử dụng dòng lệnh Terminal / CMD
```bash
# Đăng nhập với tài khoản kenny hoặc root
mysql -u kenny -p123456 < database/chatbot_tour_full.sql
```

---

## 3. Thống kê 10 bảng chuẩn hóa

1. **`users`** (3 tài khoản): `admin`, `kenny`, `user`.
2. **`categories`** (6 danh mục): Du lịch biển, Miền Bắc, Miền Trung, Miền Nam, Tây Nguyên, Sông nước miền Tây.
3. **`tours`** (20 tour du lịch): Chi tiết giá, điểm đến, thời lượng, mô tả, ảnh đại diện.
4. **`tour_schedule`** (52 lịch trình): Lịch trình tham quan từng ngày (Day 1, Day 2, Day 3...).
5. **`qa_data`** (1.290 câu hỏi đáp tri thức AI): Phục vụ mạng nơ-ron sâu PyTorch và không gian nhúng 64 chiều.
6. **`chat_sessions`** (24 phiên chat mẫu): Phiên trò chuyện của người dùng.
7. **`chat_history`** (98 tin nhắn): Lịch sử đối thoại phục vụ cơ chế AI Tự học (Continual Learning).
8. **`bookings`**: Đơn đặt tour trực tuyến.
9. **`reviews`**: Nhận xét và đánh giá 5 sao.
10. **`favorites`**: Danh sách tour yêu thích (Wishlist).
