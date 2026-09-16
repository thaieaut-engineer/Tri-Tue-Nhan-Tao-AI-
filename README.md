# Trí tuệ nhân tạo (AI)
## Đề tài: Xây Dựng Chatbot Xử Lý Ngôn Ngữ Tự Nhiên Tư Vấn Tour Du Lịch

### Danh sách các thành viên:
1. Quang Duy Thai
2. Truong Hoai Son
3. Tran Long Vu
4. Le Nguyen Nam Anh

---

### A. Chức năng người dùng (User Features)
- [x] **Đăng ký tài khoản**: Tạo tài khoản mới, mã hóa mật khẩu bảo mật (Werkzeug Security).
- [x] **Đăng nhập**: Xác thực người dùng, phân quyền theo vai trò (User/Admin).
- [x] **Đăng xuất**: Xóa phiên làm việc an toàn.
- [x] **Chatbot hỏi đáp tự nhiên**: Ứng dụng tiền xử lý tiếng Việt, TF-IDF và thuật toán Naive Bayes kết hợp Cosine Similarity để tư vấn giá tour, thời gian, lịch trình, tìm kiếm tour.
- [x] **Tra cứu thời tiết thời gian thực (OpenWeatherMap & Open-Meteo API)**: Tích hợp API thời tiết, tra cứu nhiệt độ, độ ẩm, sức gió, tình trạng mây/mưa và lời khuyên chuẩn bị cho mọi địa điểm bất kỳ (Hà Nội, TP.HCM, Đà Lạt, Phú Quốc, Sa Pa, quốc tế...). Tự động dự phòng thông minh ngay cả khi chưa có API key.
- [x] **Tìm kiếm thông tin trên Internet (Web Search)**: Tự động tìm kiếm trên mạng (DuckDuckGo, Wikipedia tiếng Việt) khi câu hỏi mở rộng về cẩm nang, ẩm thực du lịch, hoặc các địa điểm chưa có trong database nội bộ.
- [x] **Tìm kiếm tour**: Tìm kiếm đa tiêu chí theo từ khóa, lọc theo danh mục, khoảng giá và sắp xếp.
- [x] **Xem chi tiết tour**: Xem thông tin chi tiết tour và lịch trình từng ngày.
- [x] **Xem lịch sử trò chuyện**: Quản lý các phiên hội thoại, xem lại chi tiết tin nhắn hỏi đáp với AI và tiếp tục chat.

---

### B. Chức năng quản trị (Admin Features)
- [x] **Đăng nhập Admin**: Bảo vệ bằng decorator `@admin_required`.
- [x] **Thêm / sửa / xóa tour**: Quản lý danh sách tour, giá vé, hình ảnh, điểm đến và lịch trình từng ngày (`tour_schedule`).
- [x] **Quản lý danh mục tour**: Phân loại tour du lịch theo khu vực hoặc chủ đề.
- [x] **Quản lý câu hỏi mẫu & câu trả lời (AI Q&A)**: Thêm, sửa, xóa tập dữ liệu huấn luyện, phân loại intent.
- [x] **Huấn luyện lại AI (Retrain Model)**: Cập nhật trực tiếp mô hình AI từ trang quản trị không cần khởi động lại server.
- [x] **Quản lý tài khoản**: Xem danh sách người dùng, nâng/hạ quyền Admin hoặc xóa tài khoản.
- [x] **Xem lịch sử hỏi đáp**: Theo dõi toàn bộ câu hỏi và phản hồi của khách hàng trong hệ thống.

---

### C. Công nghệ sử dụng
- **Backend**: Python 3, Flask, MySQL Connector.
- **AI & NLP**: Scikit-Learn (TF-IDF Vectorizer, Multinomial Naive Bayes, Cosine Similarity), Unicodedata, RegEx.
- **Web Search**: DuckDuckGo API / DDGS, Wikipedia REST API.
- **Database**: MySQL (`chatbot_tour`).
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5, Bootstrap Icons.

---

### D. Cấu trúc thư mục dự án
```text
├── ai/                     # Xử lý ngôn ngữ tự nhiên và huấn luyện mô hình
│   ├── preprocess.py       # Chuẩn hóa tiếng Việt, xóa dấu, lọc ký tự
│   ├── train_model.py      # Huấn luyện mô hình Naive Bayes phân loại intent
│   ├── weather_service.py  # Dịch vụ tra cứu thời tiết (OpenWeatherMap & Open-Meteo)
│   ├── web_search.py       # Tìm kiếm thông tin trên mạng Internet (DuckDuckGo, Wikipedia)
│   └── chatbot.py          # Lớp Chatbot kết hợp NLP nội bộ, Thời tiết và Web Search
├── data/
│   └── sample_qa.json      # Dữ liệu câu hỏi & câu trả lời mẫu
├── database/
│   ├── db.py               # Kết nối cơ sở dữ liệu MySQL
│   └── schema.sql          # Kịch bản khởi tạo database, bảng và dữ liệu mẫu
├── models/                 # Thao tác dữ liệu (Data Access Objects)
│   ├── category.py         # Quản lý danh mục tour
│   ├── chat_history.py     # Quản lý tin nhắn hội thoại
│   ├── chat_session.py     # Quản lý phiên hội thoại
│   ├── qa_data.py          # Quản lý dữ liệu hỏi đáp AI
│   ├── tour.py             # Quản lý tour và tìm kiếm
│   ├── tour_schedule.py    # Quản lý lịch trình tour
│   └── user.py             # Quản lý người dùng, mã hóa mật khẩu
├── routes/                 # Điều hướng (Controllers)
│   ├── admin_routes.py     # Quản trị viên (/admin)
│   ├── auth_routes.py      # Đăng ký, đăng nhập, đăng xuất
│   ├── chat_routes.py      # Giao diện chat và API chatbot (/chatbot, /api/chat)
│   └── tour_routes.py      # Danh sách và chi tiết tour (/tours)
├── static/
│   ├── css/style.css       # Tùy biến giao diện
│   └── js/script.js        # Script bổ trợ
├── templates/              # Giao diện Jinja2
│   ├── admin/              # Giao diện trang quản trị
│   ├── base.html           # Layout dùng chung (Navbar, Footer)
│   ├── chatbot.html        # Giao diện Chatbot AI (hỗ trợ Web Search)
│   ├── history.html        # Lịch sử hội thoại
│   ├── index.html          # Trang chủ
│   ├── login.html          # Đăng nhập
│   ├── register.html       # Đăng ký
│   ├── tour_detail.html    # Chi tiết tour
│   └── tours.html          # Danh sách tour
├── .env                    # Biến môi trường kết nối MySQL
├── app.py                  # Điểm khởi chạy ứng dụng Flask
├── check_and_init_db.py    # Script tự động kiểm tra và khởi tạo MySQL
├── requirements.txt        # Danh sách thư viện cần thiết
└── weatherapi.py           # File cấu hình API Key thời tiết (hỗ trợ fallback khi trống)
```

---

### E. Hướng dẫn cài đặt và chạy ứng dụng

#### 1. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

#### 2. Kiểm tra và tự động khởi tạo cơ sở dữ liệu:
Chạy script kiểm tra (tự động phát hiện MySQL Docker, Windows Service, XAMPP, tự sửa password `.env`, tạo database và tài khoản admin):
```bash
python check_and_init_db.py
```

#### 3. Cấu hình thời tiết (Tùy chọn):
Nếu có API key của [OpenWeatherMap](https://openweathermap.org/), bạn điền vào file `.env`:
```env
OPENWEATHER_API_KEY=your_openweathermap_api_key_here
```
*(Nếu không điền, hệ thống sẽ tự động chuyển sang Open-Meteo Global API miễn phí hoàn toàn, không giới hạn lượt gọi).*

#### 4. Khởi chạy ứng dụng:
```bash
python app.py
```
Truy cập: `http://localhost:5000`
- **Tài khoản Quản trị viên (Admin)**: `admin` / `admin123`
- **Chatbot AI**: `/chatbot` (hỗ trợ tự động tra cứu nội bộ, tra cứu thời tiết real-time và tìm kiếm thông tin trên Internet)