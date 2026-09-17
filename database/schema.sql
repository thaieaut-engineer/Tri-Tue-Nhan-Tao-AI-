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

INSERT INTO categories (id, name, description) VALUES
(1, 'Du lịch biển', 'Các tour du lịch biển đảo cát trắng, nắng vàng'),
(2, 'Du lịch miền Bắc', 'Các tour khám phá danh thắng, văn hóa và thiên nhiên miền Bắc'),
(3, 'Du lịch miền Trung', 'Các tour di sản miền Trung, biển xanh và danh lam thắng cảnh'),
(4, 'Du lịch miền Nam', 'Các tour trải nghiệm miền Nam sầm uất và biển đảo nhiệt đới'),
(5, 'Du lịch Tây Nguyên', 'Các tour khám phá cao nguyên, thành phố sương mù và đồi chè'),
(6, 'Du lịch sông nước miền Tây', 'Các tour miệt vườn, chợ nổi và nét đẹp phù sa đồng bằng sông Cửu Long')
ON DUPLICATE KEY UPDATE name=VALUES(name), description=VALUES(description);

INSERT INTO tours
(id, category_id, name, destination, duration, price, description, image)
VALUES
(1, 3, 'Tour Đà Nẵng 3 ngày 2 đêm', 'Đà Nẵng', '3 ngày 2 đêm', 4500000, 'Khám phá Bà Nà Hills, Cầu Vàng, Bán đảo Sơn Trà, Chùa Linh Ứng và bãi biển Mỹ Khê xinh đẹp.', 'https://images.unsplash.com/photo-1559592413-7cec4d0cae2b?w=800&auto=format&fit=crop'),
(2, 1, 'Tour Nha Trang 3 ngày 2 đêm', 'Nha Trang', '3 ngày 2 đêm', 4200000, 'Thiên đường biển đảo Nha Trang, vui chơi VinWonders, lặn biển ngắm san hô tại Hòn Mun và tắm bùn khoáng.', 'https://images.unsplash.com/photo-1583417319070-4a69db38a482?w=800&auto=format&fit=crop'),
(3, 2, 'Tour Hạ Long 2 ngày 1 đêm', 'Hạ Long', '2 ngày 1 đêm', 3500000, 'Nghỉ dưỡng du thuyền 5 sao trên Vịnh Hạ Long, chèo thuyền kayak qua Hang Luồn, ngắm Hang Sửng Sốt và Đảo Ti Tốp.', 'https://images.unsplash.com/photo-1528127269322-539801943592?w=800&auto=format&fit=crop'),
(4, 1, 'Tour Phú Quốc 3 ngày 2 đêm', 'Phú Quốc', '3 ngày 2 đêm', 5200000, 'Khám phá đảo ngọc Phú Quốc, check-in Grand World thành phố không ngủ, trải nghiệm Cáp treo Hòn Thơm và lặn ngắm san hô.', 'https://images.unsplash.com/photo-1540555700478-4be289fbecef?w=800&auto=format&fit=crop'),
(5, 5, 'Tour Đà Lạt 3 ngày 2 đêm', 'Đà Lạt', '3 ngày 2 đêm', 3800000, 'Khám phá thành phố ngàn hoa Đà Lạt, chinh phục Đỉnh Langbiang, check-in Thung Lũng Tình Yêu, Đồi chè Cầu Đất và Chợ đêm.', 'https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800&auto=format&fit=crop'),
(6, 2, 'Tour Sa Pa 3 ngày 2 đêm', 'Sa Pa', '3 ngày 2 đêm', 4100000, 'Chinh phục Đỉnh Fansipan nóc nhà Đông Dương, tìm hiểu văn hóa người H''Mông tại Bản Cát Cát và ngắm Đèo Ô Quy Hồ kỳ vĩ.', 'https://images.unsplash.com/photo-1544644181-1484b3fdfc62?w=800&auto=format&fit=crop'),
(7, 3, 'Tour Quy Nhơn - Phú Yên 4 ngày 3 đêm', 'Quy Nhơn', '4 ngày 3 đêm', 4900000, 'Chiêm ngưỡng vẻ đẹp hoang sơ của Kỳ Co, Eo Gió, Ghềnh Đá Đĩa và xứ sở hoa vàng trên cỏ xanh Phú Yên.', 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800&auto=format&fit=crop'),
(8, 6, 'Tour Cần Thơ - Miền Tây 2 ngày 1 đêm', 'Cần Thơ', '2 ngày 1 đêm', 2800000, 'Trải nghiệm nét văn hóa sông nước Chợ nổi Cái Răng, thưởng thức trái cây miệt vườn Nam Bộ và du ngoạn Bến Ninh Kiều.', 'https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=800&auto=format&fit=crop')
ON DUPLICATE KEY UPDATE name=VALUES(name), destination=VALUES(destination), duration=VALUES(duration), price=VALUES(price), description=VALUES(description), image=VALUES(image);

INSERT INTO tour_schedule
(tour_id, day_number, location, activity, description)
VALUES
-- Tour 1: Đà Nẵng
(1, 1, 'Đà Nẵng', 'Đón khách, nhận phòng và tham quan trung tâm', 'Đón khách tại sân bay/ga, viếng Bán đảo Sơn Trà, Chùa Linh Ứng và tắm biển Mỹ Khê.'),
(1, 2, 'Bà Nà Hills', 'Khám phá Bà Nà Hills và Cầu Vàng', 'Đi cáp treo lên Bà Nà Hills, check-in Cầu Vàng Bàn Tay Phật, vui chơi tại Fantasy Park.'),
(1, 3, 'Hội An - Tiễn khách', 'Dạo phố cổ Hội An và mua sắm đặc sản', 'Tham quan Chùa Cầu Hội An, thả đèn hoa đăng và tiễn khách tại sân bay Đà Nẵng.'),

-- Tour 2: Nha Trang
(2, 1, 'Nha Trang', 'Đón khách, tham quan Tháp Bà Ponagar và tắm bùn', 'Tham quan tháp Chăm cổ kính, thư giãn với dịch vụ ngâm bùn khoáng nóng.'),
(2, 2, 'VinWonders Nha Trang', 'Vui chơi giải trí trọn ngày tại VinWonders', 'Trải nghiệm cáp treo vượt biển, công viên nước, vịnh phao nổi và show diễn Tata Show.'),
(2, 3, 'Vịnh Nha Trang', 'Du ngoạn 4 đảo, lặn ngắm san hô Hòn Mun', 'Đi cano cao tốc tham quan Hòn Mun, Hòn Tằm, thưởng thức tiệc trái cây và hải sản.'),

-- Tour 3: Hạ Long
(3, 1, 'Vịnh Hạ Long', 'Check-in du thuyền 5 sao, Hang Sửng Sốt', 'Thưởng thức buffet trưa trên du thuyền, chèo thuyền kayak khám phá Hang Sửng Sốt và Hang Luồn.'),
(3, 2, 'Đảo Ti Tốp', 'Tắm biển Đảo Ti Tốp, ngắm toàn cảnh vịnh', 'Leo đỉnh núi Ti Tốp ngắm toàn cảnh Vịnh Hạ Long, tham gia lớp học nấu ăn và tiễn khách.'),

-- Tour 4: Phú Quốc
(4, 1, 'Bắc Đảo Phú Quốc', 'Grand World - Thành phố không ngủ', 'Check-in dòng sông Venice lãng mạn, tham quan Bảo tàng Gấu Teddy và xem nhạc nước Tinh Hoa Việt Nam.'),
(4, 2, 'Nam Đảo Phú Quốc', 'Cáp treo Hòn Thơm và cano 4 đảo', 'Trải nghiệm cáp treo 3 dây vượt biển, cano lặn ngắm san hô tại Hòn Mây Rút và Hòn Gầm Ghì.'),
(4, 3, 'Thị trấn Dương Đông', 'Dinh Cậu, vườn tiêu và nhà thùng nước mắm', 'Tìm hiểu nghề làm nước mắm truyền thống, mua sắm ngọc trai và hải sản tươi sống.'),

-- Tour 5: Đà Lạt
(5, 1, 'Đà Lạt', 'Quảng trường Lâm Viên, Hồ Xuân Hương và Chợ đêm', 'Check-in nụ hoa Atiso khổng lồ, đạp vịt Hồ Xuân Hương, dạo chợ đêm thưởng thức bánh tráng nướng.'),
(5, 2, 'Langbiang', 'Đỉnh Langbiang và Thung Lũng Tình Yêu', 'Đi xe jeep chinh phục đỉnh Langbiang huyền thoại, check-in đồi hoa Thung Lũng Tình Yêu.'),
(5, 3, 'Cầu Đất', 'Săn mây Đồi chè Cầu Đất, Vườn dâu tây', 'Đón bình minh tại đồi chè Cầu Đất, trải nghiệm hái dâu tây tại vườn công nghệ cao.'),

-- Tour 6: Sa Pa
(6, 1, 'Sa Pa', 'Nhà thờ đá và Bản Cát Cát', 'Xe limousine đưa đón từ Hà Nội lên Sa Pa, tham quan bản Cát Cát tìm hiểu văn hóa H''Mông.'),
(6, 2, 'Fansipan', 'Chinh phục Đỉnh Fansipan nóc nhà Đông Dương', 'Đi cáp treo 3 dây hiện đại lên đỉnh Fansipan 3.143m, chiêm bái quần thể tâm linh trên mây.'),
(6, 3, 'Đèo Ô Quy Hồ', 'Cổng trời Ô Quy Hồ và Thác Bạc', 'Ngắm hoàng hôn trên tứ đại đỉnh đèo Ô Quy Hồ, thưởng thức cá hồi Sa Pa và mua quà đặc sản.'),

-- Tour 7: Quy Nhơn - Phú Yên
(7, 1, 'Quy Nhơn', 'Kỳ Co - Eo Gió', 'Đi cano ra bãi Kỳ Co tắm biển nước trong xanh, ngắm hoàng hôn rực rỡ tại Eo Gió.'),
(7, 2, 'Phú Yên', 'Ghềnh Đá Đĩa và Bãi Xép Hoa vàng cỏ xanh', 'Chiêm ngưỡng kiệt tác địa chất Ghềnh Đá Đĩa, chụp ảnh tại phim trường Tôi thấy hoa vàng trên cỏ xanh.'),
(7, 3, 'Quy Nhơn', 'Tháp Bánh Ít và Chùa Thiên Hưng', 'Khám phá văn hóa Chăm pa tại Tháp Bánh Ít, viếng cảnh chùa Thiên Hưng thanh tịnh.'),
(7, 4, 'Quy Nhơn', 'Mua sắm đặc sản và tiễn khách', 'Thưởng thức bánh hỏi lòng heo, mua nem chợ Huyện, tré Bình Định làm quà.'),

-- Tour 8: Cần Thơ - Miền Tây
(8, 1, 'Cần Thơ', 'Nhà cổ Bình Thủy và Bến Ninh Kiều', 'Tham quan nhà cổ kiến trúc Pháp - Hoa, du ngoạn du thuyền Bến Ninh Kiều ngắm sông Hậu.'),
(8, 2, 'Cái Răng', 'Chợ nổi Cái Răng và Miệt vườn trái cây', 'Đi thuyền lúc sáng sớm trải nghiệm văn hóa chợ nổi Cái Răng, thưởng thức hủ tiếu nổi và ăn trái cây tại vườn.');