CREATE DATABASE chatbot_tour
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE chatbot_tour;

CREATE DATABASE IF NOT EXISTS chatbot_tour
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE chatbot_tour;

-- =========================================
-- 1. BẢNG NGƯỜI DÙNG
-- =========================================
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    role ENUM('user', 'admin') DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =========================================
-- 2. BẢNG DANH MỤC TOUR
-- =========================================
CREATE TABLE IF NOT EXISTS categories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =========================================
-- 3. BẢNG TOUR
-- =========================================
CREATE TABLE IF NOT EXISTS tours (
    id INT AUTO_INCREMENT PRIMARY KEY,
    category_id INT,
    name VARCHAR(200) NOT NULL,
    destination VARCHAR(200) NOT NULL,
    duration VARCHAR(50) NOT NULL,
    price DECIMAL(12,2) NOT NULL,
    description TEXT,
    image VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (category_id)
        REFERENCES categories(id)
        ON DELETE SET NULL
        ON UPDATE CASCADE
);

-- =========================================
-- 4. BẢNG LỊCH TRÌNH TOUR
-- =========================================
CREATE TABLE IF NOT EXISTS tour_schedule (
    id INT AUTO_INCREMENT PRIMARY KEY,
    tour_id INT NOT NULL,
    day_number INT NOT NULL,
    location VARCHAR(200) NOT NULL,
    activity VARCHAR(255) NOT NULL,
    description TEXT,

    FOREIGN KEY (tour_id)
        REFERENCES tours(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- =========================================
-- 5. BẢNG DỮ LIỆU HỎI ĐÁP AI
-- =========================================
CREATE TABLE IF NOT EXISTS qa_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    intent VARCHAR(100) NOT NULL,
    tour_id INT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (tour_id)
        REFERENCES tours(id)
        ON DELETE SET NULL
        ON UPDATE CASCADE
);

-- =========================================
-- 6. BẢNG PHIÊN TRÒ CHUYỆN
-- =========================================
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    title VARCHAR(200) DEFAULT 'Cuộc trò chuyện mới',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- =========================================
-- 7. BẢNG LỊCH SỬ TRÒ CHUYỆN
-- =========================================
CREATE TABLE IF NOT EXISTS chat_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id)
        REFERENCES chat_sessions(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- =========================================
-- DỮ LIỆU MẪU
-- =========================================

INSERT INTO categories (name, description) VALUES
('Du lịch biển', 'Các tour du lịch biển'),
('Du lịch miền Bắc', 'Các tour du lịch miền Bắc'),
('Du lịch miền Trung', 'Các tour du lịch miền Trung'),
('Du lịch miền Nam', 'Các tour du lịch miền Nam');

INSERT INTO tours
(category_id, name, destination, duration, price, description)
VALUES
(3, 'Tour Đà Nẵng 3 ngày 2 đêm',
 'Đà Nẵng', '3 ngày 2 đêm', 4500000,
 'Khám phá Đà Nẵng, Bà Nà Hills và biển Mỹ Khê'),

(1, 'Tour Nha Trang 3 ngày 2 đêm',
 'Nha Trang', '3 ngày 2 đêm', 4200000,
 'Tham quan biển Nha Trang và các địa điểm nổi tiếng'),

(2, 'Tour Hạ Long 2 ngày 1 đêm',
 'Hạ Long', '2 ngày 1 đêm', 3500000,
 'Khám phá vịnh Hạ Long và các địa điểm xung quanh');

INSERT INTO tour_schedule
(tour_id, day_number, location, activity, description)
VALUES
(1, 1, 'Đà Nẵng', 'Đón khách và tham quan thành phố',
 'Khởi hành và tham quan các địa điểm trung tâm'),

(1, 2, 'Bà Nà Hills', 'Tham quan và vui chơi',
 'Khám phá Bà Nà Hills'),

(1, 3, 'Biển Mỹ Khê', 'Tắm biển và kết thúc tour',
 'Tham quan biển Mỹ Khê');

INSERT INTO qa_data (question, answer, intent, tour_id) VALUES
('Tour Đà Nẵng giá bao nhiêu?',
 'Giá tour Đà Nẵng 3 ngày 2 đêm là 4.500.000đ.',
 'hoi_gia', 1),

('Tôi muốn đi Đà Nẵng',
 'Bạn có thể tham khảo Tour Đà Nẵng 3 ngày 2 đêm.',
 'tim_tour', 1),

('Tour Đà Nẵng đi mấy ngày?',
 'Tour Đà Nẵng có thời gian 3 ngày 2 đêm.',
 'hoi_thoi_gian', 1),

('Tour Đà Nẵng có những địa điểm nào?',
 'Tour tham quan Đà Nẵng, Bà Nà Hills và biển Mỹ Khê.',
 'hoi_lich_trinh', 1),

('Xin chào',
 'Xin chào! Tôi có thể giúp bạn tìm tour du lịch phù hợp.',
 'chao_hoi', NULL),

('Tạm biệt',
 'Cảm ơn bạn đã sử dụng chatbot. Chúc bạn có chuyến đi vui vẻ!',
 'tam_biet', NULL);