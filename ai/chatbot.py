"""
ai/chatbot.py
Module Chatbot AI thông minh thế hệ mới kết hợp 3 kỹ thuật cốt lõi:
 1. TF-IDF Vectorizer (1-3 ngrams, sublinear_tf): Biểu diễn ngữ nghĩa không gian vector đa chiều.
 2. Complement / Multinomial Naive Bayes: Mô hình học máy phân loại ý định (Intent Classification) chính xác cao.
 3. Hybrid Intent-Weighted Cosine Similarity: Thuật toán so khớp tương đồng kết hợp xác suất ý định.
 4. Multi-turn Conversational Memory: Bộ nhớ ngữ cảnh hội thoại đa lượt duy trì thực thể điểm đến.
 5. Smart Recommendation Engine: Trích xuất thực thể ngân sách, thời lượng và sở thích để gợi ý tour thông minh.
 6. Live API Weather Service & Directed Web Search: Tra cứu thời tiết thời gian thực và tìm kiếm Internet có định hướng.
"""

import os
import re
import json
import time
from sklearn.metrics.pairwise import cosine_similarity

from database.db import get_connection
from ai.preprocess import preprocess_text, remove_accents, correct_travel_typos
from ai.train_model import train_model
from ai.web_search import search_web_for_travel, format_web_response
from ai.weather_service import (
    is_weather_query,
    extract_city_from_question,
    get_weather_for_location,
    format_weather_response
)


# ====================================================================
# DANH MỤC ĐỊA DANH MỞ RỘNG VÀ TỪ KHÓA CHỦ ĐỀ
# ====================================================================
OUTSIDE_DESTINATIONS = [
    # Quốc tế / Nước ngoài
    "nước ngoài", "quốc tế", "ngoại quốc", "thái lan", "nhật bản", "hàn quốc", "trung quốc",
    "châu âu", "châu á", "mỹ", "hoa kỳ", "singapore", "malaysia", "đài loan", "bali", "úc",
    "pháp", "anh", "đức", "ý", "nga", "campuchia", "lào", "dubai", "hồng kông", "ấn độ",
    # Các điểm du lịch Việt Nam khác ngoài 20 tour hệ thống
    "hải phòng", "hà nội", "sài gòn", "hồ chí minh", "tphcm", "tam đảo",
    "mai châu", "ba bể", "quảng bình", "phong nha", "kẻ bàng", "bến tre",
    "bạc liêu", "cà mau", "bình định", "quảng ninh", "đồ sơn", "bạch long vĩ",
    "lý sơn", "bình ba", "nam du", "phú thọ"
]

AMBIGUOUS_SHORT_COUNTRIES = {
    "ý": r'\b(?:nước|du lịch|tour|đi|vé|đến)\s+ý\b|\bitalia\b|\bitaly\b',
    "anh": r'\b(?:nước|du lịch|tour|đi|vé|đến)\s+anh\b|\bvương quốc anh\b|\bengland\b|\buk\b',
    "mỹ": r'(?<!biển\s)(?<!bãi\s)\b(?:nước|du lịch|tour|đi|vé|đến)\s+mỹ\b|\bhoa kỳ\b|\busa\b',
    "úc": r'\b(?:nước|du lịch|tour|đi|vé|đến)\s+úc\b|\baustralia\b',
    "pháp": r'(?<!làng\s)\b(?:nước|du lịch|tour|đi|vé|đến)\s+pháp\b|\bfrance\b',
    "đức": r'\b(?:nước|du lịch|tour|đi|vé|đến)\s+đức\b|\bgermany\b',
    "lào": r'\b(?:nước|du lịch|tour|đi|vé|đến)\s+lào\b|\blaopdr\b',
    "nga": r'\b(?:nước|du lịch|tour|đi|vé|đến)\s+nga\b|\brussia\b',
}

TRAVEL_TOPIC_KEYWORDS = [
    "thời tiết", "nhiệt độ", "có mưa không", "mùa nào đẹp", "tháng mấy nên đi", "mùa bão",
    "món ăn", "ẩm thực", "đặc sản", "quán ngon", "ăn gì", "chơi gì", "uống gì", "cafe đẹp",
    "khách sạn", "homestay", "resort", "nhà nghỉ", "chỗ ở", "booking",
    "kinh nghiệm", "cẩm nang", "mẹo du lịch", "chuẩn bị gì", "mang gì", "lưu ý",
    "vé máy bay", "tàu hỏa", "xe khách", "cách đi đến", "phương tiện", "thuê xe",
    "lễ hội", "sự kiện", "bắn pháo hoa", "check in", "sống ảo", "địa điểm đẹp"
]


# ====================================================================
# CƠ SỞ TRI THỨC ĐIỂM ĐẾN & ẨM THỰC BẢN ĐỊA (LOCAL DESTINATION TIPS)
# ====================================================================
LOCAL_DESTINATION_TIPS = {
    "đà nẵng": {
        "food": "Mì Quảng ếch Bếp Trang, Bánh tráng cuốn thịt heo hai đầu da Trần, Hải sản tươi sống Bé Mặn, Bánh xèo tôm nhảy Năm Hiền, Chè sầu Liên.",
        "best_time": "Từ tháng 3 đến tháng 8: trời trong xanh, nắng rực rỡ, biển Mỹ Khê êm đềm thích hợp tắm biển và vui chơi Bà Nà Hills.",
        "pack_tips": "Nên mang trang phục bơi, kem chống nắng, mũ rộng vành khi đi biển và áo khoác nhẹ vì Bà Nà Hills trên núi cao có tiết trời se lạnh."
    },
    "nha trang": {
        "food": "Bún chả cá sứa Loan, Nem nướng Ninh Hòa Đặng Văn Quyên, Bò nướng Lạc Cảnh, Hải sản tươi sống làng chài, Bánh căn mực.",
        "best_time": "Từ tháng 1 đến tháng 8: vịnh Nha Trang nước trong vắt êm đềm, thời tiết nắng vàng rực rỡ lý tưởng để du ngoạn đảo và tắm bùn.",
        "pack_tips": "Trang phục biển thoải mái, kính bơi, túi chống nước bảo vệ điện thoại khi tham gia lặn ngắm san hô tại Hòn Mun."
    },
    "hạ long": {
        "food": "Chả mực giã tay Hạ Long, Bún cù kỳ bến Đoan, Sá sùng xào tỏi, Bánh gật gù Tiên Yên, Sam biển xào chua ngọt.",
        "best_time": "Tháng 4 - tháng 6 và tháng 9 - tháng 11: khí hậu mát mẻ, nắng nhẹ, rất thuận lợi cho hành trình du thuyền ngủ đêm trên vịnh.",
        "pack_tips": "Chuẩn bị trang phục năng động để chèo kayak, giày thể thao đế mềm chống trơn khi leo hang Sửng Sốt và kính râm."
    },
    "phú quốc": {
        "food": "Gỏi cá trích Mai Hương, Bún quậy Kiến Xây trứ danh, Nhum biển nướng mỡ hành, Ghẹ Hàm Ninh hấp bia, Rượu sim rừng Bảy Gáo.",
        "best_time": "Từ cuối tháng 10 đến tháng 4 năm sau (mùa khô biển lặng, sóng êm, nước biển màu ngọc bích tuyệt đẹp để đi cano 4 đảo).",
        "pack_tips": "Mang trang phục rực rỡ chụp ảnh check-in Sunset Town / Grand World, kính râm, kem chống nắng thân thiện môi trường biển."
    },
    "đà lạt": {
        "food": "Lẩu gà lá é Tao Ngộ, Bánh tráng nướng Dì Đinh, Lẩu bò Quán Gỗ Ba Toa, Bánh ướt lòng gà Long, Sữa đậu nành Tăng Bạt Hổ, Kem bơ Thanh Thảo.",
        "best_time": "Từ tháng 11 đến tháng 4 năm sau (mùa khô se lạnh, ngàn hoa đua nở, mùa cỏ hồng và mùa săn mây đồi chè Cầu Đất đỉnh cao).",
        "pack_tips": "Thời tiết se lạnh 14-18°C về đêm, bạn nên chuẩn bị áo ấm, khăn quàng cổ nhẹ, ô gấp gọn và giày thể thao êm chân."
    },
    "sa pa": {
        "food": "Thắng cố ngựa A Quỳnh, Lẩu cá hồi cá tầm Fansipan, Lợn cắp nách nướng than hoa, Cơm lam gà đồi nướng mắc khén, Rau mầm đá chấm trứng dầm.",
        "best_time": "Tháng 9 - 10 (mùa lúa chín vàng óng ả Mường Hoa) hoặc tháng 12 - 2 (mùa đông mây băng tuyết và mùa hoa mận hoa đào trắng rừng).",
        "pack_tips": "Nhiệt độ trên đỉnh Fansipan (3.143m) rất thấp và nhiều gió, hãy chuẩn bị áo phao ấm dày, găng tay, mũ len và giày leo núi chống trơn."
    },
    "quy nhơn": {
        "food": "Bánh hỏi cháo lòng Diên Hồng, Bánh xèo tôm nhảy Gia Vỹ, Chả ram tôm đất, Cua huỳnh đế hấp nước dừa, Bún chả cá Ngọc Liên.",
        "best_time": "Từ tháng 3 đến tháng 8: biển Kỳ Co trong vắt màu ngọc bích, trời nắng đẹp rạng rỡ, thích hợp tắm biển và lặn ngắm san hô.",
        "pack_tips": "Kỳ Co và Eo Gió đón gió biển lộng, bạn nên mang kính râm, kem chống nắng SPF cao, mũ có quai buộc và giày sandals chống nước."
    },
    "cần thơ": {
        "food": "Lẩu mắm Dạ Lý, Bánh cống cô Út, Cá lóc nướng trui cuốn lá sen non, Bánh xèo củ hủ dừa Mười Xiềm, Trái cây tươi ngọt hái tại vườn.",
        "best_time": "Từ tháng 9 đến tháng 11 (mùa nước nổi đặc trưng Nam Bộ) hoặc từ tháng 5 đến tháng 8 (mùa các loại trái cây miệt vườn chín rộ).",
        "pack_tips": "Trang phục gọn nhẹ thoáng mát; nên dậy sớm từ 5h00 sáng để xuống bến thuyền đón trọn khoảnh khắc nhộn nhịp nhất của Chợ nổi Cái Răng."
    },
    "hà giang": {
        "food": "Bánh tam giác mạch nướng than, Thắng dền phố cổ Đồng Văn, Cháo ấu tẩu ấm bụng, Thịt trâu gác bếp, Rượu ngô men lá Quản Bạ.",
        "best_time": "Tháng 9 - 10 (mùa lúa chín vàng Hoàng Su Phì) và tháng 10 - 12 (mùa hoa tam giác mạch phủ hồng khắp các sườn núi đá vôi).",
        "pack_tips": "Cung đường đèo dốc uốn lượn hùng vĩ (Mã Pí Lèng, Dốc Thẩm Mã), bạn nên chuẩn bị thuốc chống say xe, áo khoác gió ấm và máy ảnh."
    },
    "ninh bình": {
        "food": "Thịt dê núi Ninh Bình nướng tảng chao dầu, Cơm cháy chấm sốt dê thơm lừng, Gỏi cá nhệch Kim Sơn, Ốc núi luộc sả, Miến lươn Bà Phấn.",
        "best_time": "Mùa xuân (tháng 1 - 3) trẩy hội Chùa Bái Đính và du xuân Tràng An; hoặc cuối tháng 5 - đầu tháng 6 ngắm lúa chín vàng Tam Cốc.",
        "pack_tips": "Chuẩn bị giày thể thao để leo 500 bậc đá Hang Múa ngắm toàn cảnh Tam Cốc và trang phục kín đáo lịch sự khi viếng Chùa Bái Đính."
    },
    "mộc châu": {
        "food": "Bê chao Mộc Châu thơm mềm, Cá suối chiên giòn, Cải ngồng luộc chấm trứng dầm, Ốc đá Suối Bàng, Sữa tươi và dâu tây Chimi Farm.",
        "best_time": "Tháng 1 - 2 (mùa hoa mận, hoa mơ nở trắng rừng cao nguyên) và tháng 10 - 12 (mùa hoa cải trắng, hoa dã quỳ vàng rực rỡ).",
        "pack_tips": "Khí hậu cao nguyên trong lành se lạnh; bạn nên chuẩn bị áo khoác nhẹ, giày bệt hoặc giày thể thao để dạo bước giữa các đồi chè."
    },
    "cát bà": {
        "food": "Tu hài nướng mỡ hành béo ngậy, Bề bề rang muối, Sam 7 món Cát Bà, Mực một nắng xào cần tỏi, Cá song hấp gừng xì dầu.",
        "best_time": "Từ tháng 4 đến tháng 9: thời điểm lý tưởng nhất để tắm biển Đảo Khỉ, chèo kayak Vịnh Lan Hạ và tận hưởng làn gió biển mát rượi.",
        "pack_tips": "Mang đồ bơi, túi chống nước cho điện thoại khi chèo kayak hang Sáng Tối trên Vịnh Lan Hạ và thuốc say sóng nếu đi tàu cao tốc."
    },
    "huế": {
        "food": "Bún bò Huế Mụ Rơi / bà Búp, Cơm hến Đập Đá, Bánh bèo - nậm - lọc bà Đỏ, Nem lụi Hoàng Triều, Chè bột lọc bọc heo quay.",
        "best_time": "Từ tháng 1 đến tháng 4: thời tiết Cố đô dịu mát, nắng nhẹ, cảnh sắc trăm hoa khoe sắc rất đẹp để tham quan các lăng tẩm.",
        "pack_tips": "Mặc trang phục nhã nhặn, kín đáo (áo dài truyền thống rất đẹp khi check-in) khi tham quan Đại Nội Hoàng Thành và Lăng Khải Định."
    },
    "hội an": {
        "food": "Cao lầu Thanh, Cơm gà Bà Buội, Bánh mì Phượng / Madam Khánh, Hoành thánh chiên Vạn Lộc, Nước thảo mộc Mót Hội An thanh mát.",
        "best_time": "Từ tháng 2 đến tháng 7: mùa khô ráo nắng đẹp, đặc biệt vào đêm rằm 14-15 âm lịch phố cổ tắt đèn điện thả hoa đăng lung linh.",
        "pack_tips": "Phố cổ cấm xe máy ban đêm nên bạn hãy chuẩn bị giày êm để tản bộ, trang phục vintage chụp ảnh tuyệt đẹp dưới ánh đèn lồng."
    },
    "buôn ma thuột": {
        "food": "Bún đỏ phố cổ Phan Đình Giót, Gà nướng Bản Đôn - Cơm lam muối ớt, Gỏi cà đắng cá cơm, Cà phê Robusta Tây Nguyên nguyên chất.",
        "best_time": "Từ tháng 12 đến tháng 4 (mùa khô Tây Nguyên thời tiết mát mẻ, mùa hoa cà phê nở trắng muốt bạt ngàn các sườn đồi).",
        "pack_tips": "Mang giày thể thao chống trơn khi trekking ngắm Thác Dray Nur, trang phục gọn gàng thoải mái khi chèo thuyền độc mộc Hồ Lắk."
    },
    "phan thiết": {
        "food": "Lẩu thả Mũi Né, Bánh canh chả cá cô Xí, Răng mực nướng Loan, Dông cát nướng muối ớt, Hải sản tươi ngon bờ kè Mũi Né.",
        "best_time": "Từ tháng 11 đến tháng 4 năm sau: mùa trời trong xanh, ít mưa, sóng biển êm và đồi cát Bàu Trắng rực rỡ dưới nắng vàng.",
        "pack_tips": "Khu vực đồi cát Bàu Trắng nhiều nắng và gió cát, hãy mang kính mát, kem chống nắng, khăn choàng và trang phục nổi bật đi xe jeep."
    },
    "vũng tàu": {
        "food": "Bánh khọt Gốc Vú Sữa / Cô Ba Vũng Tàu, Lẩu cá đuối Trương Công Định, Bông lan trứng muối Gột Cột Cờ, Gỏi cá mai Ba Hưng, Hải sản Gành Hào.",
        "best_time": "Quanh năm (chỉ cách TP.HCM 2 giờ di chuyển), đặc biệt đẹp nhất từ tháng 11 đến tháng 4 (biển êm, sóng nhẹ, nắng ráo).",
        "pack_tips": "Chuẩn bị giày thể thao leo gần 800 bậc đá lên Tượng Chúa Kito Vua và trang phục tắm biển Bãi Sau năng động."
    },
    "an giang": {
        "food": "Bún cá lóc Châu Đốc, Bò bảy món Núi Sam Tư Thiêng, Lẩu cá linh bông điên điển (mùa nước nổi), Bánh bò thốt nốt, Tung lò mò bò Chăm.",
        "best_time": "Từ tháng 9 đến tháng 11 (mùa nước nổi rừng tràm Trà Sư bạt ngàn bèo xanh) hoặc dịp Lễ hội Vía Bà Chúa Xứ (tháng 4 âm lịch).",
        "pack_tips": "Chuẩn bị nón lá hoặc mũ che nắng khi ngồi xuồng ba lá lướt thảm bèo Rừng tràm Trà Sư, trang phục trang nghiêm khi viếng Miếu Bà."
    },
    "côn đảo": {
        "food": "Cháo hàu Côn Đảo, Cua mặt trăng hấp bia, Ốc vú nàng luộc xả, Mực một nắng nướng sa tế, Mứt hạt bàng rang muối đường đặc sản.",
        "best_time": "Từ tháng 3 đến tháng 9: thời điểm biển êm sóng lặng nhất trong năm dù có mưa rào ngắn, thích hợp tham quan và lặn biển san hô.",
        "pack_tips": "Chuẩn bị trang phục lịch sự, kín đáo (áo dài tay, quần sẫm màu) khi làm lễ viếng Mộ Cô Sáu tại Nghĩa trang Hàng Dương lúc nửa đêm."
    }
}

# ====================================================================
# LỊCH TRÌNH TOUR FALLBACK ĐẦY ĐỦ 52 NGÀY CHO 20 TOUR NỘI BỘ
# ====================================================================
FALLBACK_TOUR_SCHEDULES = {
    1: [
        {"day_number": 1, "location": "Đà Nẵng", "activity": "Đón khách, nhận phòng và tham quan trung tâm", "description": "Đón khách tại sân bay/ga, viếng Bán đảo Sơn Trà, Chùa Linh Ứng và tắm biển Mỹ Khê."},
        {"day_number": 2, "location": "Bà Nà Hills", "activity": "Khám phá Bà Nà Hills và Cầu Vàng", "description": "Đi cáp treo lên Bà Nà Hills, check-in Cầu Vàng Bàn Tay Phật, vui chơi tại Fantasy Park."},
        {"day_number": 3, "location": "Hội An - Tiễn khách", "activity": "Dạo phố cổ Hội An và mua sắm đặc sản", "description": "Tham quan Chùa Cầu Hội An, thả đèn hoa đăng và tiễn khách tại sân bay Đà Nẵng."}
    ],
    2: [
        {"day_number": 1, "location": "Nha Trang", "activity": "Đón khách, tham quan Tháp Bà Ponagar và tắm bùn", "description": "Đón khách, tham quan Tháp Bà Ponagar linh thiêng và trải nghiệm tắm bùn khoáng nóng thư giãn."},
        {"day_number": 2, "location": "VinWonders Nha Trang", "activity": "Vui chơi giải trí trọn ngày tại VinWonders", "description": "Đi cáp treo vượt biển đến đảo Hòn Tre, trải nghiệm vòng đu quay khổng lồ và công viên nước."},
        {"day_number": 3, "location": "Vịnh Nha Trang", "activity": "Du ngoạn 4 đảo, lặn ngắm san hô Hòn Mun", "description": "Đi cano khám phá Hòn Mun, Hòn Tằm ngắm rạn san hô rực rỡ và tiễn khách ra sân bay Cam Ranh."}
    ],
    3: [
        {"day_number": 1, "location": "Hạ Long - Du thuyền", "activity": "Check-in du thuyền 5 sao, tham quan Hang Sửng Sốt", "description": "Lên du thuyền 5 sao vịnh Hạ Long, ăn trưa buffet hải sản, khám phá Hang Sửng Sốt và chèo kayak."},
        {"day_number": 2, "location": "Đảo Ti Tốp - Hà Nội", "activity": "Leo đỉnh Ti Tốp ngắm toàn cảnh vịnh, trở về Hà Nội", "description": "Tập thái cực quyền đón bình minh trên boong tàu, tắm biển Ti Tốp rồi xe đưa đoàn về Hà Nội."}
    ],
    4: [
        {"day_number": 1, "location": "Bắc Đảo Phú Quốc", "activity": "Grand World - Thành phố không ngủ", "description": "Check-in dòng sông Venice lãng mạn, tham quan Bảo tàng Gấu Teddy và xem nhạc nước Tinh Hoa Việt Nam."},
        {"day_number": 2, "location": "Nam Đảo Phú Quốc", "activity": "Cáp treo Hòn Thơm và cano 4 đảo", "description": "Trải nghiệm cáp treo 3 dây vượt biển, cano lặn ngắm san hô tại Hòn Mây Rút và Hòn Gầm Ghì."},
        {"day_number": 3, "location": "Thị trấn Dương Đông", "activity": "Dinh Cậu, vườn tiêu và nhà thùng nước mắm", "description": "Tìm hiểu nghề làm nước mắm truyền thống, mua sắm ngọc trai và hải sản tươi sống."}
    ],
    5: [
        {"day_number": 1, "location": "Đà Lạt", "activity": "Đón khách, check-in Quảng trường Lâm Viên và Ga Đà Lạt", "description": "Đón khách, chụp ảnh nụ hoa Atiso biểu tượng, tham quan Ga xe lửa cổ kính và ngắm Hồ Xuân Hương."},
        {"day_number": 2, "location": "Langbiang - Thung Lũng Tình Yêu", "activity": "Chinh phục Đỉnh Langbiang và ngắm ngàn hoa", "description": "Đi xe jeep lên đỉnh Langbiang ngắm trọn thung lũng, dạo Thung Lũng Tình Yêu rực rỡ sắc hoa."},
        {"day_number": 3, "location": "Đồi chè Cầu Đất - Tiễn khách", "activity": "Săn mây Đồi chè Cầu Đất, Chùa Linh Phước", "description": "Đón bình minh săn biển mây bồng bềnh tại Cầu Đất, viếng Chùa Ve Chai độc đáo trước khi ra sân bay."}
    ],
    6: [
        {"day_number": 1, "location": "Sa Pa", "activity": "Nhà thờ đá và Bản Cát Cát", "description": "Xe limousine đưa đón từ Hà Nội lên Sa Pa, tham quan bản Cát Cát tìm hiểu văn hóa H'Mông."},
        {"day_number": 2, "location": "Fansipan", "activity": "Chinh phục Đỉnh Fansipan nóc nhà Đông Dương", "description": "Đi cáp treo 3 dây hiện đại lên đỉnh Fansipan 3.143m, chiêm bái quần thể tâm linh trên mây."},
        {"day_number": 3, "location": "Đèo Ô Quy Hồ", "activity": "Cổng trời Ô Quy Hồ và Thác Bạc", "description": "Ngắm hoàng hôn trên tứ đại đỉnh đèo Ô Quy Hồ, thưởng thức cá hồi Sa Pa và mua quà đặc sản."}
    ],
    7: [
        {"day_number": 1, "location": "Quy Nhơn", "activity": "Đón khách, KDL Ghềnh Ráng Tiên Sa, Mộ Hàn Mặc Tử", "description": "Đón khách tại sân bay Phù Cát, tham quan bãi tắm Hoàng Hậu đá trứng và viếng thi sĩ Hàn Mặc Tử."},
        {"day_number": 2, "location": "Kỳ Co - Eo Gió", "activity": "Cano cao tốc đi Kỳ Co, ngắm hoàng hôn Eo Gió", "description": "Tắm biển làn nước trong vắt tại bãi Kỳ Co, lặn ngắm san hô Bãi Dứa và check-in cung đường Eo Gió."},
        {"day_number": 3, "location": "Phú Yên", "activity": "Khám phá Ghềnh Đá Đĩa, Nhà thờ Mằng Lăng, Bãi Xép", "description": "Chiêm ngưỡng kỳ quan núi đá bazan hình lục giác Ghềnh Đá Đĩa, phim trường 'Tôi thấy hoa vàng trên cỏ xanh'."},
        {"day_number": 4, "location": "Tháp Đôi - Tiễn khách", "activity": "Tháp Đôi Chăm Pa và mua sắm đặc sản", "description": "Tìm hiểu di tích tháp Chăm Tháp Đôi cổ kính, mua chả ram tôm đất, bánh ít lá gai trước khi ra sân bay."}
    ],
    8: [
        {"day_number": 1, "location": "Cần Thơ", "activity": "Nhà cổ Bình Thủy, Vườn cò Bằng Lăng, Bến Ninh Kiều", "description": "Khám phá nhà cổ kiến trúc Pháp 150 năm tuổi, ngắm hàng vạn cánh cò bay về tổ và du thuyền Bến Ninh Kiều."},
        {"day_number": 2, "location": "Chợ nổi Cái Răng", "activity": "Thuyền xuôi Chợ nổi Cái Răng, Miệt vườn cây trái", "description": "Trải nghiệm văn hóa buôn bán ghe xuồng trên sông từ sáng sớm, thưởng thức hủ tiếu nổi và trái cây miệt vườn."}
    ],
    9: [
        {"day_number": 1, "location": "Hà Nội - Quản Bạ - Yên Minh", "activity": "Ngắm Núi Đôi Quản Bạ, Rừng thông Yên Minh, check-in Dốc Thẩm Mã.", "description": "Vượt Dốc Bắc Sum, chiêm ngưỡng Núi Đôi Quản Bạ và rừng thông Yên Minh ngút ngàn xanh mát."},
        {"day_number": 2, "location": "Đồng Văn - Mã Pí Lèng - Nho Quế", "activity": "Chinh phục Đèo Mã Pí Lèng, đi thuyền ngắm Hẻm Tu Sản trên sông Nho Quế, dạo Phố cổ Đồng Văn.", "description": "Chinh phục một trong tứ đại đỉnh đèo hiểm trở nhất VN, đi thuyền dưới hẻm vực Tu Sản sâu nhất Đông Nam Á."},
        {"day_number": 3, "location": "Lũng Cú - Dinh Vua Mèo - Hà Nội", "activity": "Chào cờ tại Cột cờ Quốc gia Lũng Cú, thăm Dinh Thự Họ Vương (Vua Mèo) và trở về Hà Nội.", "description": "Chạm tay vào cột mốc cực Bắc thiêng liêng của Tổ Quốc, tìm hiểu kiến trúc Dinh Thự Vua Mèo rồi về Hà Nội."}
    ],
    10: [
        {"day_number": 1, "location": "Hà Nội - Tràng An - Hang Múa", "activity": "Đi thuyền nan khám phá Hang Sáng Hang Tối Quần thể Tràng An, leo đỉnh Hang Múa.", "description": "Thuyền nan lướt nhẹ qua các hang động kỳ ảo Tràng An di sản thế giới, leo 500 bậc đá Hang Múa ngắm Tam Cốc."},
        {"day_number": 2, "location": "Chùa Bái Đính - Hà Nội", "activity": "Viếng Chùa Bái Đính chiêm bái đại tượng Phật bằng đồng lớn nhất Đông Nam Á.", "description": "Hành hương quần thể chùa Bái Đính nguy nga, thưởng thức đặc sản thịt dê cơm cháy Ninh Bình rồi về Hà Nội."}
    ],
    11: [
        {"day_number": 1, "location": "Hà Nội - Mộc Châu - Đồi Chè Trái Tim", "activity": "Vượt Đèo Thung Khe mây phủ, check-in Đồi chè trái tim xanh mướt, Rừng thông Bản Áng.", "description": "Dạo bước trên những luống chè xanh bát ngát, chèo thuyền ngắm cảnh hồ rừng thông Bản Áng mộng mơ."},
        {"day_number": 2, "location": "Thác Dải Yếm - Cầu Kính Bạch Long - Hà Nội", "activity": "Tham quan Thác Dải Yếm hùng vĩ, thử thách Cầu kính Bạch Long dài nhất thế giới.", "description": "Ngắm thác nước đổ bọt trắng xóa, trải nghiệm cầu kính đi bộ lập kỷ lục Guinness rồi khởi hành về Hà Nội."}
    ],
    12: [
        {"day_number": 1, "location": "Hải Phòng - Đảo Cát Bà - Vịnh Lan Hạ", "activity": "Du thuyền khám phá Vịnh Lan Hạ hoang sơ, chèo kayak Hang Sáng Hang Tối, tắm biển Đảo Khỉ.", "description": "Lên du thuyền ngắm hàng trăm đảo đá vôi kỳ vĩ, tự do chèo thuyền kayak luồn lách qua các vòm hang nước ngập."},
        {"day_number": 2, "location": "Vườn Quốc Gia Cát Bà - Hà Nội / Hải Phòng", "activity": "Trekking đỉnh Ngự Lâm ngắm trọn rừng kim giao, tham quan Làng chài cổ Cái Bèo nghìn năm.", "description": "Khám phá thảm thực vật Vườn quốc gia Cát Bà, tìm hiểu làng chài nổi cổ nhất Việt Nam trước khi tiễn khách."}
    ],
    13: [
        {"day_number": 1, "location": "Đại Nội Huế - Chùa Thiên Mụ", "activity": "Tham quan Ngọ Môn, Điện Thái Hòa, Tử Cấm Thành, viếng Chùa Thiên Mụ cổ kính bên bờ sông Hương.", "description": "Lắng nghe câu chuyện lịch sử 13 đời vua triều Nguyễn, chiêm bái tháp Phước Duyên 7 tầng soi bóng dòng sông Hương."},
        {"day_number": 2, "location": "Lăng Khải Định - Nghe Ca Huế Sông Hương", "activity": "Khám phá Lăng Khải Định đỉnh cao kiến trúc sành sứ, tối đi thuyền rồng nghe ca Huế.", "description": "Chiêm ngưỡng bức tranh phù điêu sành sứ và bức bích họa 'Cửu Long Ẩn Vân', đêm nghe nhã nhạc ca Huế thả hoa đăng."}
    ],
    14: [
        {"day_number": 1, "location": "Rừng dừa Bảy Mẫu - Phố cổ Hội An", "activity": "Ngồi thuyền thúng lắc lư Rừng dừa Bảy Mẫu Cẩm Thanh, chiều tối bách bộ phố cổ thả hoa đăng sông Hoài.", "description": "Thử cảm giác múa thúng xoay tròn điêu luyện, tối thả hoa đăng lung linh trên sông Hoài ngắm phố đèn lồng."},
        {"day_number": 2, "location": "Cù Lao Chàm - Lặn ngắm san hô", "activity": "Cano cao tốc đưa khách ra Cù Lao Chàm, lặn ngắm rạn san hô Bãi Chồng và thưởng thức hải sản đảo.", "description": "Cano lướt sóng ra khu dự trữ sinh quyển thế giới Cù Lao Chàm, lặn ngắm san hô và thưởng thức cua đá đảo."}
    ],
    15: [
        {"day_number": 1, "location": "Buôn Ma Thuột - Bảo tàng Thế Giới Cà Phê", "activity": "Đón khách, check-in Bảo tàng Cà phê Thế Giới mang kiến trúc nhà dài Tây Nguyên, thưởng thức cafe nguyên chất.", "description": "Khám phá không gian văn hóa cà phê độc đáo, ngắm hiện vật của đồng bào Ê Đê và uống cà phê Robusta thượng hạng."},
        {"day_number": 2, "location": "Thác Dray Nur - Buôn Đôn", "activity": "Chiêm ngưỡng dòng thác Dray Nur hùng vĩ bọt tung trắng xóa, ghé Buôn Đôn tìm hiểu văn hóa săn bắt voi rừng.", "description": "Tận mắt ngắm thác Dray Nur huyền thoại với dải nước khổng lồ gầm vang, giao lưu văn hóa tại Buôn Đôn bên sông Sêrêpôk."},
        {"day_number": 3, "location": "Hồ Lắk - Buôn Jun - Tiễn khách", "activity": "Chèo thuyền độc mộc lướt trên mặt Hồ Lắk thơ mộng, khám phá buôn làng người M'Nông trước khi ra sân bay.", "description": "Lướt thuyền độc mộc giữa mặt nước mênh mông phẳng lặng của hồ nước ngọt tự nhiên lớn nhất Tây Nguyên rồi tiễn khách."}
    ],
    16: [
        {"day_number": 1, "location": "TP.HCM - Phan Thiết - Suối Tiên - Làng Chài", "activity": "Di chuyển cao tốc đến Mũi Né, lội dòng suối cát đỏ Suối Tiên, ngắm cảnh thuyền nan tấp nập tại Làng chài Mũi Né.", "description": "Lội dòng Suối Tiên tuyệt mỹ hai bên vách đất sét đỏ rực, ngắm ghe thuyền cập bến tấp nập tại làng chài truyền thống."},
        {"day_number": 2, "location": "Bàu Trắng - Đồi Cát Bay - TP.HCM", "activity": "Đón bình minh tại Bàu Trắng bằng xe địa hình mạo hiểm, trượt cát Đồi Cát Bay và mua nước mắm cá cơm.", "description": "Trải nghiệm xe jeep vượt đồi cát trắng Bàu Trắng ngắm hồ sen mát mắt, trượt ván cát tại Đồi Cát Đỏ trước khi về Sài Gòn."}
    ],
    17: [
        {"day_number": 1, "location": "TP.HCM - Bãi Sau Vũng Tàu - Tượng Chúa Kito", "activity": "Đón khách từ Sài Gòn, tắm biển Bãi Sau sóng êm, chiều leo núi nhỏ chinh phục cánh tay Tượng Chúa Kito.", "description": "Đoàn tắm biển thỏa thích tại Bãi Sau bãi cát vàng mịn màng, leo 800 bậc đá lên cánh tay Chúa Kito ngắm toàn cảnh Vũng Tàu."},
        {"day_number": 2, "location": "Ngọn Hải Đăng - Bến Thuyền Marina - TP.HCM", "activity": "Chụp ảnh tại Ngọn Hải Đăng cổ kính Pháp, ghé bến thuyền buồm Marina sang trọng và ăn bánh khọt Gốc Vú Sữa.", "description": "Ngắm thành phố từ ngọn hải đăng lâu đời nhất Đông Nam Á, check-in bến thuyền buồm rực rỡ và thưởng thức bánh khọt nóng giòn."}
    ],
    18: [
        {"day_number": 1, "location": "TP.HCM - Sa Đéc - Châu Đốc - Núi Sam", "activity": "Ghé Làng hoa Sa Đéc ngàn hoa khoe sắc, chiều đến Châu Đốc viếng Miếu Bà Chúa Xứ Núi Sam cầu tài lộc.", "description": "Dạo bước giữa muôn sắc hoa Làng hoa Sa Đéc, viếng trung tâm hành hương tâm linh Miếu Bà Chúa Xứ Núi Sam nức tiếng linh thiêng."},
        {"day_number": 2, "location": "Rừng Tràm Trà Sư - TP.HCM", "activity": "Đi xuồng ba lá len lỏi trong rừng tràm Trà Sư ngắm đàn chim trời bay lượn trên thảm bèo xanh mướt mát mắt.", "description": "Ngồi xuồng ba lá lướt trên tấm thảm bèo cám xanh rì rực rỡ dưới ánh nắng rọi qua kẽ lá tràm, chụp ảnh cầu tre vạn bước."}
    ],
    19: [
        {"day_number": 1, "location": "Đón sân bay Cỏ Ống - Bãi Nhát - Mũi Cá Mập", "activity": "Đón khách tại sân bay Côn Đảo, ngắm đỉnh Tình Yêu, Bãi Nhát, tối 23h00 viếng Mộ Cô Sáu Nghĩa trang Hàng Dương.", "description": "Xe đón đoàn, chiêm ngưỡng thiên nhiên hoang sơ Bãi Nhát, đêm 23h00 làm lễ dâng hoa viếng mộ liệt nữ Võ Thị Sáu linh thiêng."},
        {"day_number": 2, "location": "Nhà tù Côn Đảo - Bãi Đầm Trầu", "activity": "Tham quan Trại giam Phú Hải, Chuồng Cọp Pháp - Mỹ lịch sử, chiều thư giãn tắm biển Bãi Đầm Trầu ngắm máy bay hạ cánh.", "description": "Nghe thuyết minh xúc động về ý chí kiên cường của các chiến sĩ cách mạng, chiều tắm biển Bãi Đầm Trầu cát trắng mịn màng."},
        {"day_number": 3, "location": "Miếu Bà Phi Yến - Chợ Côn Đảo - Tiễn sân bay", "activity": "Viếng Miếu Bà Phi Yến, ghé chợ mua mứt hạt bàng đặc sản và xe tiễn ra sân bay Cỏ Ống.", "description": "Viếng An Sơn Miếu tưởng nhớ thứ phi Hoàng Phi Yến, mua sắm hạt bàng rang muối gừng rồi lên chuyến bay trở về."}
    ],
    20: [
        {"day_number": 1, "location": "Check-in Resort 5 Sao - Grand World", "activity": "Xe đón sân bay về resort 5 sao nhận phòng view biển, chiều tối tham quan Grand World và xem show Tinh Hoa Việt Nam.", "description": "Nghỉ dưỡng thượng lưu tại resort 5 sao bờ biển, thưởng thức ẩm thực cao cấp, dạo chơi Grand World xem nhạc nước Venice."},
        {"day_number": 2, "location": "Vinpearl Safari - Công viên VinWonders", "activity": "Khám phá Vườn thú bán hoang dã Safari lớn nhất Việt Nam, vui chơi thỏa thích công viên chủ đề VinWonders.", "description": "Ngồi xe bus chuyên dụng ngắm hổ Bengal, tê giác tại Safari và thỏa sức phiêu lưu tàu lượn siêu tốc tại VinWonders."},
        {"day_number": 3, "location": "Cáp Treo Hòn Thơm - Cano 4 Đảo VIP", "activity": "Đi cáp treo vượt biển Hòn Thơm, cano đưa đoàn lặn biển san hô Hòn Gầm Ghì, flycam quay chụp kỷ niệm.", "description": "Trải nghiệm cáp treo 3 dây lập kỷ lục Guinness, cano cao tốc lặn biển seawalker ngắm rạn san hô nguyên sơ đầy màu sắc."},
        {"day_number": 4, "location": "Thị Trấn Hoàng Hôn - Mua sắm ngọc trai - Tiễn khách", "activity": "Check-in Cầu Hôn (Kiss Bridge), mua ngọc trai Phú Quốc và xe đưa đoàn ra sân bay trở về.", "description": "Chiêm ngưỡng kiệt tác Cầu Hôn bên bờ biển Địa Trung Hải Sunset Town, mua ngọc trai chất lượng cao trước khi bay về."}
    ]
}


# ====================================================================
# BỘ NHỚ NGỮ CẢNH HỘI THOẠI ĐA LƯỢT (MULTI-TURN CONVERSATION MEMORY)
# ====================================================================
class ChatSessionMemory:
    """
    Duy trì trạng thái ngữ cảnh trò chuyện (Active Tour, Điểm đến, Ý định gần nhất)
    giúp người dùng hỏi tiếp các câu nối tiếp mà không cần lặp lại tên tour.
    """
    def __init__(self, ttl_seconds=3600):
        self._sessions = {}
        self.ttl_seconds = ttl_seconds

    def get_context(self, session_id):
        if not session_id:
            return {}
        sid = str(session_id)
        data = self._sessions.get(sid)
        if not data:
            return {}
        if time.time() - data.get("updated_at", 0) > self.ttl_seconds:
            del self._sessions[sid]
            return {}
        return data

    def update_context(self, session_id, **kwargs):
        if not session_id:
            return
        sid = str(session_id)
        if sid not in self._sessions:
            self._sessions[sid] = {"created_at": time.time(), "turns": 0}
        self._sessions[sid].update(kwargs)
        self._sessions[sid]["updated_at"] = time.time()
        self._sessions[sid]["turns"] += 1


# ====================================================================
# BỘ TRÍCH XUẤT THỰC THỂ NGÂN SÁCH VÀ THỜI LƯỢNG (NER)
# ====================================================================
def extract_budget(text):
    """Trích xuất số tiền ngân sách từ câu hỏi (ví dụ: '3tr5', '4 triệu', '500k', '4.000.000đ')."""
    t_low = text.lower()

    # Mẫu 3tr5 hoặc 4 triệu 2
    m1 = re.search(r'(\d+)\s*(?:tr|triệu|trieu)\s*(\d+)', t_low)
    if m1:
        trieu = int(m1.group(1))
        le = int(m1.group(2))
        if le < 10:
            return trieu * 1_000_000 + le * 100_000
        elif le < 100:
            return trieu * 1_000_000 + le * 10_000
        else:
            return trieu * 1_000_000 + le * 1_000

    # Mẫu 4.5 triệu hoặc 4tr hoặc 500k
    m2 = re.search(r'(\d+[\.,]?\d*)\s*(triệu|trieu|tr|k|nghìn|nghin)', t_low)
    if m2:
        val = float(m2.group(1).replace(',', '.'))
        unit = m2.group(2)
        if unit in ('triệu', 'trieu', 'tr'):
            return int(val * 1_000_000)
        elif unit in ('k', 'nghìn', 'nghin'):
            return int(val * 1_000)

    # Mẫu số tiền đầy đủ: 4.000.000 hoặc 4000000
    m3 = re.search(r'(\d{1,3}(?:[\.,]\d{3})+)', t_low)
    if m3:
        clean_num = re.sub(r'[\.,]', '', m3.group(1))
        return int(clean_num)

    return None


def extract_duration_days(text):
    """Trích xuất số ngày mong muốn (2 ngày, 3N2Đ, 4 ngày...)."""
    m = re.search(r'(\d+)\s*(?:ngày|ngay|n\d*đ)', text.lower())
    if m:
        return int(m.group(1))
    return None


# ====================================================================
# LỚP CHATBOT AI CHÍNH
# ====================================================================
class Chatbot:

    def __init__(self):
        self.vectorizer = None   # Mô hình TF-IDF
        self.model = None        # Mô hình DeepHybridModel (Deep MLP + ComplementNB)
        self.data_vectors = None # Vector TF-IDF của tập dữ liệu mẫu

        self.questions = []
        self.answers = []
        self.intents = []
        self.tour_ids = []
        self.local_tours = []
        self.tour_schedules = {}
        self.memory = ChatSessionMemory()
        self.last_predicted_intent = None
        self.last_confidence = None
        self.reload()

    def reload(self):
        """
        Nạp lại dữ liệu và huấn luyện lại mô hình AI (TF-IDF + Naive Bayes).
        """
        self.questions = []
        self.answers = []
        self.intents = []
        self.tour_ids = []
        self.local_tours = []
        self.tour_schedules = {}

        # 1. Nạp dữ liệu câu hỏi đáp và danh sách tour
        self.load_data()
        self.load_local_tours()
        self.load_tour_schedules()

        # 2. Huấn luyện TF-IDF Vectorizer và Naive Bayes Classifier
        self.vectorizer, self.model = train_model()

        # 3. Tiền tính toán ma trận vector TF-IDF cho toàn bộ câu hỏi mẫu
        if self.vectorizer and self.questions:
            self.data_vectors = self.vectorizer.transform(self.questions)

    def load_local_tours(self):
        """Lấy danh sách các tour và điểm đến trong hệ thống."""
        connection = get_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("SELECT id, name, destination, price, duration, description FROM tours")
                self.local_tours = cursor.fetchall()
            except Exception as e:
                print("Lỗi nạp local tours:", e)
            finally:
                cursor.close()
                connection.close()

        if not self.local_tours:
            self.local_tours = [
                {"id": 1, "name": "Tour Đà Nẵng 3 ngày 2 đêm", "destination": "Đà Nẵng", "price": 4500000, "duration": "3 ngày 2 đêm", "description": "Khám phá Bà Nà Hills, Cầu Vàng, Bán đảo Sơn Trà, Chùa Linh Ứng và biển Mỹ Khê."},
                {"id": 2, "name": "Tour Nha Trang 3 ngày 2 đêm", "destination": "Nha Trang", "price": 4200000, "duration": "3 ngày 2 đêm", "description": "Trải nghiệm VinWonders, lặn biển ngắm san hô tại Hòn Mun, tắm bùn khoáng."},
                {"id": 3, "name": "Tour Hạ Long 2 ngày 1 đêm", "destination": "Hạ Long", "price": 3500000, "duration": "2 ngày 1 đêm", "description": "Du thuyền 5 sao vịnh Hạ Long, chèo thuyền kayak, tham quan hang Sửng Sốt."},
                {"id": 4, "name": "Tour Phú Quốc 3 ngày 2 đêm", "destination": "Phú Quốc", "price": 5200000, "duration": "3 ngày 2 đêm", "description": "Khám phá đảo ngọc Phú Quốc, check-in Grand World, Cáp treo Hòn Thơm và lặn ngắm san hô."},
                {"id": 5, "name": "Tour Đà Lạt 3 ngày 2 đêm", "destination": "Đà Lạt", "price": 3800000, "duration": "3 ngày 2 đêm", "description": "Thành phố ngàn hoa Đà Lạt, chinh phục Đỉnh Langbiang, Thung Lũng Tình Yêu, Đồi chè Cầu Đất."},
                {"id": 6, "name": "Tour Sa Pa 3 ngày 2 đêm", "destination": "Sa Pa", "price": 4100000, "duration": "3 ngày 2 đêm", "description": "Chinh phục đỉnh Fansipan nóc nhà Đông Dương, tìm hiểu văn hóa bản Cát Cát và ngắm đèo Ô Quy Hồ."},
                {"id": 7, "name": "Tour Quy Nhơn - Phú Yên 4 ngày 3 đêm", "destination": "Quy Nhơn", "price": 4900000, "duration": "4 ngày 3 đêm", "description": "Khám phá Kỳ Co, Eo Gió, Ghềnh Đá Đĩa và xứ sở hoa vàng trên cỏ xanh Phú Yên."},
                {"id": 8, "name": "Tour Cần Thơ - Miền Tây 2 ngày 1 đêm", "destination": "Cần Thơ", "price": 2800000, "duration": "2 ngày 1 đêm", "description": "Trải nghiệm văn hóa chợ nổi Cái Răng, thưởng thức trái cây miệt vườn Nam Bộ và Bến Ninh Kiều."},
                {"id": 9, "name": "Tour Hà Giang 3 ngày 2 đêm", "destination": "Hà Giang", "price": 3200000, "duration": "3 ngày 2 đêm", "description": "Chinh phục Cột cờ Lũng Cú cực Bắc Tổ quốc, Đèo Mã Pí Lèng hiểm trở, đi thuyền ngắm Hẻm Tu Sản và dòng sông Nho Quế xanh ngọc bích."},
                {"id": 10, "name": "Tour Ninh Bình 2 ngày 1 đêm", "destination": "Ninh Bình", "price": 1900000, "duration": "2 ngày 1 đêm", "description": "Khám phá Quần thể danh thắng Tràng An di sản thế giới, viếng Chùa Bái Đính lớn nhất Đông Nam Á, chinh phục đỉnh Hang Múa ngắm Tam Cốc."},
                {"id": 11, "name": "Tour Mộc Châu 2 ngày 1 đêm", "destination": "Mộc Châu", "price": 1650000, "duration": "2 ngày 1 đêm", "description": "Cao nguyên Mộc Châu xanh mướt với Đồi chè trái tim, Rừng thông Bản Áng, Thác Dải Yếm hùng vĩ và mùa hoa mận hoa mơ trắng rừng."},
                {"id": 12, "name": "Tour Cát Bà - Vịnh Lan Hạ 2 ngày 1 đêm", "destination": "Cát Bà", "price": 2400000, "duration": "2 ngày 1 đêm", "description": "Khám phá đảo ngọc Cát Bà, du thuyền ngoạn cảnh Vịnh Lan Hạ hoang sơ, chèo kayak Hang Sáng Hang Tối và tắm biển tại Đảo Khỉ."},
                {"id": 13, "name": "Tour Huế - Cố Đô Di Sản 2 ngày 1 đêm", "destination": "Huế", "price": 2600000, "duration": "2 ngày 1 đêm", "description": "Tham quan Quần thể Di tích Cố đô Huế: Đại Nội Hoàng Thành, Lăng Khải Định, Chùa Thiên Mụ cổ kính và đi thuyền nghe ca Huế trên sông Hương."},
                {"id": 14, "name": "Tour Hội An - Cù Lao Chàm 2 ngày 1 đêm", "destination": "Hội An", "price": 2950000, "duration": "2 ngày 1 đêm", "description": "Dạo bước trong lòng Phố cổ Hội An lung linh đèn lồng, đi cano siêu tốc ra đảo Cù Lao Chàm lặn ngắm san hô và trải nghiệm Rừng dừa Bảy Mẫu."},
                {"id": 15, "name": "Tour Buôn Ma Thuột - Khám Phá Tây Nguyên 3 ngày 2 đêm", "destination": "Buôn Ma Thuột", "price": 3300000, "duration": "3 ngày 2 đêm", "description": "Chinh phục Thác Dray Nur cuồn cuộn, cưỡi voi Buôn Đôn, chèo thuyền độc mộc trên Hồ Lắk và check-in Bảo tàng Cà phê Thế Giới độc đáo."},
                {"id": 16, "name": "Tour Phan Thiết - Mũi Né 2 ngày 1 đêm", "destination": "Phan Thiết", "price": 2500000, "duration": "2 ngày 1 đêm", "description": "Trải nghiệm xe jeep vượt Đồi Cát Trắng Bàu Trắng, lội Suối Tiên kỳ vĩ, khám phá Làng chài Mũi Né và thưởng thức hải sản biển tươi ngon."},
                {"id": 17, "name": "Tour Vũng Tàu Biển Xanh 2 ngày 1 đêm", "destination": "Vũng Tàu", "price": 1850000, "duration": "2 ngày 1 đêm", "description": "Tắm biển Bãi Sau, chinh phục Tượng Chúa Kito Vua ngắm trọn thành phố biển, check-in Ngọn Hải Đăng cổ và Bến thuyền buồm Marina."},
                {"id": 18, "name": "Tour Rừng Tràm Trà Sư - An Giang 2 ngày 1 đêm", "destination": "An Giang", "price": 2200000, "duration": "2 ngày 1 đêm", "description": "Tắc ráng lướt trên thảm bèo xanh mướt Rừng Tràm Trà Sư, viếng Miếu Bà Chúa Xứ Núi Sam linh thiêng và ghé thăm Làng hoa Sa Đéc rực rỡ."},
                {"id": 19, "name": "Tour Côn Đảo Tâm Linh & Nghỉ Dưỡng 3 ngày 2 đêm", "destination": "Côn Đảo", "price": 6200000, "duration": "3 ngày 2 đêm", "description": "Hành trình linh thiêng viếng Mộ Cô Sáu tại Nghĩa trang Hàng Dương, tham quan Trại giam Chuồng Cọp, nghỉ dưỡng tại bãi biển Đầm Trầu hoang sơ tuyệt đẹp."},
                {"id": 20, "name": "Tour Phú Quốc Nghỉ Dưỡng 5 Sao 4 ngày 3 đêm", "destination": "Phú Quốc", "price": 7500000, "duration": "4 ngày 3 đêm", "description": "Trải nghiệm đẳng cấp tại resort 5 sao bờ biển, vé vui chơi VinWonders và vườn thú bán hoang dã Safari, cáp treo Hòn Thơm và cano 4 đảo VIP."}
            ]

    def load_data(self):
        """Đọc dữ liệu câu hỏi mẫu từ MySQL hoặc fallback file sample_qa.json."""
        loaded_from_db = False
        connection = get_connection()

        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("SELECT id, question, answer, intent, tour_id FROM qa_data")
                data = cursor.fetchall()

                if data and len(data) > 0:
                    for row in data:
                        self.questions.append(preprocess_text(row["question"]))
                        self.answers.append(row["answer"])
                        self.intents.append(row["intent"])
                        self.tour_ids.append(row["tour_id"])
                    loaded_from_db = True
            except Exception as e:
                print("Lỗi load dữ liệu chatbot từ DB:", e)
            finally:
                cursor.close()
                connection.close()

        if not loaded_from_db or len(self.questions) == 0:
            json_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_qa.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        samples = json.load(f)
                        for item in samples:
                            self.questions.append(preprocess_text(item["question"]))
                            self.answers.append(item["answer"])
                            self.intents.append(item["intent"])
                            self.tour_ids.append(item.get("tour_id"))
                except Exception as e:
                    print("Lỗi nạp file sample_qa.json:", e)

    def detect_destination_context(self, question):
        """Nhận diện thực thể điểm đến (Entity Recognition)."""
        q_low = question.lower()

        local_keywords_map = {
            "đà nẵng": ["đà nẵng", "da nang", "bà nà", "ba na", "mỹ khê", "sơn trà", "cầu rồng", "linh ứng"],
            "nha trang": ["nha trang", "vinpearl", "vinwonders", "tháp bà", "hòn tằm", "hòn mun"],
            "hạ long": ["hạ long", "ha long", "tuần châu", "vịnh hạ long", "hang sửng sốt", "ti tốp", "du thuyền hạ long"],
            "phú quốc": ["phú quốc", "phu quoc", "hòn thơm", "grand world", "sunset town", "bãi sao", "dinh cậu"],
            "đà lạt": ["đà lạt", "da lat", "langbiang", "thung lũng tình yêu", "đồi chè cầu đất", "hồ xuân hương"],
            "sa pa": ["sa pa", "sapa", "fansipan", "cát cát", "ô quy hồ", "hàm rồng"],
            "quy nhơn": ["quy nhơn", "quy nhon", "kỳ co", "eo gió", "ghềnh đá đĩa", "phú yên"],
            "cần thơ": ["cần thơ", "can tho", "cái răng", "chợ nổi", "bến ninh kiều", "miền tây"],
            "hà giang": ["hà giang", "ha giang", "mã pí lèng", "ma pi leng", "lũng cú", "lung cu", "sông nho quế", "nho que", "tu sản", "đồng văn"],
            "ninh bình": ["ninh bình", "ninh binh", "tràng an", "trang an", "bái đính", "bai dinh", "hang múa", "hang mua", "tam cốc", "tam coc"],
            "mộc châu": ["mộc châu", "moc chau", "bản áng", "ban ang", "thác dải yếm", "dải yếm", "đồi chè trái tim", "thung khe"],
            "cát bà": ["cát bà", "cat ba", "vịnh lan hạ", "lan ha", "đảo khỉ", "dao khi", "cái bèo"],
            "huế": ["huế", "cố đô huế", "đại nội", "lăng khải định", "thiên mụ", "sông hương", "ca huế", "chợ đông ba"],
            "hội an": ["hội an", "hoi an", "phố cổ hội an", "cù lao chàm", "cu lao cham", "rừng dừa bảy mẫu", "chùa cầu"],
            "buôn ma thuột": ["buôn ma thuột", "buon ma thuot", "đắk lắk", "dak lak", "dray nur", "hồ lắk", "ho lak", "buôn đôn", "buon don", "tây nguyên"],
            "phan thiết": ["phan thiết", "phan thiet", "mũi né", "mui ne", "bàu trắng", "bau trang", "suối tiên", "đồi cát bay"],
            "vũng tàu": ["vũng tàu", "vung tau", "bãi sau", "bai sau", "tượng chúa kito", "hải đăng vũng tàu", "bến thuyền marina"],
            "an giang": ["an giang", "rừng tràm trà sư", "trà sư", "tra su", "núi sam", "nui sam", "bà chúa xứ", "châu đốc", "sa đéc"],
            "côn đảo": ["côn đảo", "con dao", "mộ cô sáu", "cô sáu", "hàng dương", "hang duong", "chuồng cọp", "đầm trầu", "cỏ ống"]
        }

        matched_local = []
        for tour in self.local_tours:
            dest = tour["destination"].lower()
            kws = local_keywords_map.get(dest, [dest])
            found = False
            for kw in kws:
                if re.search(rf'(?:\b|^){re.escape(kw)}(?:\b|$)', q_low):
                    found = True
                    break
            if found:
                matched_local.append(tour)

        matched_outside = []
        q_unaccent = remove_accents(q_low)

        # 1. Kiểm tra các quốc gia tên ngắn dễ trùng từ vựng tiếng Việt (ý, úc, nga, anh, mỹ, pháp, đức, lào)
        for out_dest, pat in AMBIGUOUS_SHORT_COUNTRIES.items():
            if re.search(pat, q_low):
                matched_outside.append(out_dest)

        # 2. Kiểm tra các địa danh khác ngoài hệ thống với biên từ (\b)
        for out_dest in OUTSIDE_DESTINATIONS:
            if out_dest in AMBIGUOUS_SHORT_COUNTRIES:
                continue
            esc = re.escape(out_dest)
            if re.search(rf'(?:\b|^){esc}(?:\b|$)', q_low):
                matched_outside.append(out_dest)
            else:
                esc_unaccent = re.escape(remove_accents(out_dest))
                if len(esc_unaccent) > 2 and re.search(rf'(?:\b|^){esc_unaccent}(?:\b|$)', q_unaccent):
                    matched_outside.append(out_dest)

        return matched_local, matched_outside

    def compute_hybrid_cosine_similarity(self, question_vector, intent_probs, allowed_tour_ids=None):
        """
        THUẬT TOÁN SO KHỚP KẾT HỢP (HYBRID INTENT-WEIGHTED COSINE SIMILARITY):
        Kết hợp ma trận xác suất từ DeepHybridModel (Deep MLP + Naive Bayes) với độ đo Cosine TF-IDF:
        - Tính độ đo Cosine thuần túy giữa vector câu hỏi và các câu hỏi mẫu.
        - Tăng trọng số (boost) lên đến +25% cho các câu hỏi mẫu có intent trùng khớp với
          phân phối xác suất từ mô hình học sâu.
        - Lọc ưu tiên theo allowed_tour_ids nếu có điểm đến cụ thể.
        """
        if self.data_vectors is None or self.vectorizer is None or len(self.questions) == 0:
            return -1, 0.0

        raw_similarities = cosine_similarity(question_vector, self.data_vectors)[0]

        # Áp dụng trọng số tăng cường từ DeepHybridModel Intent Probabilities
        weighted_similarities = raw_similarities.copy()
        if intent_probs:
            for i, intent in enumerate(self.intents):
                prob = intent_probs.get(intent, 0.0)
                # Tăng tối đa 25% điểm cho câu hỏi mẫu đúng intent được dự đoán
                weighted_similarities[i] = raw_similarities[i] * (1.0 + 0.25 * prob)

        if allowed_tour_ids:
            valid_indices = [
                i for i, tid in enumerate(self.tour_ids)
                if tid is None or tid in allowed_tour_ids
            ]
            if valid_indices:
                filtered_sims = weighted_similarities[valid_indices]
                best_sub_idx = int(filtered_sims.argmax())
                best_index = valid_indices[best_sub_idx]
                best_score = float(weighted_similarities[best_index])
                return best_index, best_score

        best_index = int(weighted_similarities.argmax())
        best_score = float(weighted_similarities[best_index])

        return best_index, best_score

    def get_tour(self, tour_id):
        for t in self.local_tours:
            if t["id"] == tour_id:
                return t

        connection = get_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("SELECT id, name, destination, duration, price, description FROM tours WHERE id = %s", (tour_id,))
                return cursor.fetchone()
            except Exception as e:
                print("Lỗi lấy tour:", e)
            finally:
                cursor.close()
                connection.close()
        return None

    def load_tour_schedules(self):
        """Nạp danh sách lịch trình tour chi tiết từng ngày từ tour_schedule hoặc fallback."""
        connection = get_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("SELECT tour_id, day_number, location, activity, description FROM tour_schedule ORDER BY tour_id, day_number")
                rows = cursor.fetchall()
                self.tour_schedules = {}
                for r in rows:
                    self.tour_schedules.setdefault(r["tour_id"], []).append(r)
            except Exception as e:
                print("Lỗi nạp tour_schedule:", e)
            finally:
                cursor.close()
                connection.close()

        if not self.tour_schedules:
            self.tour_schedules = dict(FALLBACK_TOUR_SCHEDULES)

    def get_tour_schedule(self, tour_id):
        """Lấy danh sách lịch trình theo ngày của một tour."""
        if not self.tour_schedules:
            self.load_tour_schedules()
        return self.tour_schedules.get(tour_id, [])

    def format_rich_tour_briefing(self, tour, focus="overview"):
        """
        ĐỘNG CƠ BIÊN SOẠN BÁO CÁO TƯ VẤN TOUR CHUYÊN SÂU (EXPERT TOUR CONSULTATION BRIEFING)
        Xây dựng phản hồi tư vấn viên du lịch đầy đủ, có chiều sâu, trải nghiệm chân thực.
        - focus="schedule": Ưu tiên chi tiết lịch trình từng ngày, hoạt động và ẩm thực
        - focus="price": Ưu tiên bảng giá, dịch vụ đã bao gồm trong giá và tóm tắt ngày
        - focus="overview": Tư vấn tổng quan toàn diện (hành trình + lịch trình + ẩm thực + dịch vụ + cẩm nang)
        """
        schedules = self.get_tour_schedule(tour["id"])
        dest_key = tour.get("destination", "").lower().strip()
        tips = LOCAL_DESTINATION_TIPS.get(dest_key, {})

        lines = []

        if focus == "price":
            lines.append(f"💰 **BÁO GIÁ & DỊCH VỤ TRỌN GÓI: {tour['name']}**")
            lines.append(f"📍 Điểm đến: **{tour['destination']}** | ⏱ Thời gian: **{tour['duration']}**")
            lines.append(f"💵 **Giá trọn gói niêm yết: {tour['price']:,.0f} VNĐ/khách**\n")
            lines.append("✨ **Giá tour đã bao gồm trọn gói tất cả các dịch vụ:**")
            lines.append("  • 🏨 Nghỉ dưỡng tại khách sạn/resort tiêu chuẩn sạch sẽ, tiện nghi")
            lines.append("  • 🚗 Xe du lịch máy lạnh đời mới chất lượng cao đưa đón trọn hành trình")
            lines.append("  • 🍽️ Đầy đủ các bữa ăn chính mang hương vị đặc sản địa phương + buffet sáng")
            lines.append("  • 🎫 Toàn bộ vé tham quan danh thắng, vé thuyền/cano/cáp treo theo chương trình")
            lines.append("  • 👨‍💼 Hướng dẫn viên chuyên nghiệp, nhiệt tình, am hiểu văn hóa bản địa")
            lines.append("  • 🛡️ Bảo hiểm du lịch với hạn mức tối đa 50.000.000 VNĐ/vụ & nước suối đóng chai\n")

            if schedules:
                lines.append("🗓️ **Tóm tắt hành trình khám phá:**")
                for s in schedules:
                    lines.append(f"  • **Ngày {s['day_number']} ({s['location']})**: {s['activity']}")
                lines.append("")

            if tips.get("pack_tips"):
                lines.append(f"🎒 **Lưu ý & Chuẩn bị:** {tips['pack_tips']}\n")

            lines.append(f"👉 Bạn có thể xem hình ảnh và thông tin chi tiết của tour tại: [{tour['name']}](/tours/{tour['id']})")
            return "\n".join(lines)

        # Mặc định (focus="schedule" hoặc focus="overview")
        icon = "🗓️" if focus == "schedule" else "🌟"
        lines.append(f"{icon} **TƯ VẤN HÀNH TRÌNH: {tour['name']}**")
        lines.append(f"📍 Điểm đến: **{tour['destination']}** | ⏱ Thời lượng: **{tour['duration']}** | 💰 Giá trọn gói: **{tour['price']:,.0f} VNĐ/khách**\n")
        lines.append(f"📝 *{tour['description']}*\n")

        if schedules:
            lines.append("🗓️ **Lịch trình trải nghiệm chi tiết từng ngày:**")
            for s in schedules:
                lines.append(f"  • **Ngày {s['day_number']} ({s['location']})**: {s['activity']}")
                desc = s.get("description", "").strip()
                if desc and desc != s["activity"]:
                    lines.append(f"    - *Chi tiết*: {desc}")
            lines.append("")
        else:
            lines.append(f"🗓️ Tour được thiết kế tối ưu trong {tour['duration']} giúp bạn khám phá trọn vẹn danh thắng {tour['destination']}.\n")

        if tips.get("food"):
            lines.append(f"🍜 **Ẩm thực đặc sản không thể bỏ lỡ tại {tour['destination']}:**\n  {tips['food']}\n")

        if tips.get("best_time"):
            lines.append(f"🌤️ **Thời điểm du lịch lý tưởng nhất:** {tips['best_time']}\n")

        if tips.get("pack_tips"):
            lines.append(f"🎒 **Gợi ý chuẩn bị & trang phục:**\n  {tips['pack_tips']}\n")

        lines.append("✨ **Dịch vụ & tiện ích trọn gói bao gồm:**")
        lines.append("  • 🏨 Khách sạn/resort tiêu chuẩn đầy đủ tiện nghi")
        lines.append("  • 🚗 Xe du lịch đưa đón máy lạnh đời mới")
        lines.append("  • 🍽️ Các bữa ăn đặc sản theo chương trình")
        lines.append("  • 🎫 Vé tham quan các điểm đến theo lịch trình")
        lines.append("  • 👨‍💼 Hướng dẫn viên bản địa chu đáo suốt tuyến")
        lines.append("  • 🛡️ Bảo hiểm du lịch tối đa 50.000.000 VNĐ & nước suối đóng chai\n")

        lines.append(f"👉 Bạn có thể xem hình ảnh và chi tiết tour tại: [{tour['name']}](/tours/{tour['id']})")
        return "\n".join(lines)

    def recommend_tours_by_criteria(self, text):
        """
        ĐỘNG CƠ TƯ VẤN THÔNG MINH (SMART RECOMMENDATION ENGINE):
        Tự động bóc tách ngân sách, thời gian, sở thích và vùng miền để đề xuất danh sách tour tối ưu.
        """
        t_low = text.lower()
        budget = extract_budget(text)
        duration_days = extract_duration_days(text)

        is_cheap_query = any(w in t_low for w in ["rẻ nhất", "re nhat", "tiết kiệm", "tiet kiem", "thấp nhất", "gia re", "giá rẻ"])
        is_beach = any(w in t_low for w in ["biển", "bien", "đảo", "dao", "tắm biển", "tam bien", "lan bien", "lặn biển"])
        is_mountain = any(w in t_low for w in ["núi", "nui", "vùng cao", "vung cao", "săn mây", "san may", "tuyết", "tây bắc", "tay bac", "đèo", "deo"])
        is_central = any(w in t_low for w in ["miền trung", "mien trung", "di sản", "cố đô", "co do"])
        is_south = any(w in t_low for w in ["miền tây", "mien tay", "sông nước", "song nuoc", "miệt vườn", "chợ nổi"])
        is_north = any(w in t_low for w in ["miền bắc", "mien bac", "gần hà nội", "gan ha noi"])

        candidates = list(self.local_tours)

        # Lọc theo sở thích vùng miền
        if is_beach:
            beach_dests = ["Đà Nẵng", "Nha Trang", "Hạ Long", "Phú Quốc", "Quy Nhơn", "Cát Bà", "Phan Thiết", "Vũng Tàu", "Côn Đảo"]
            candidates = [t for t in candidates if t["destination"] in beach_dests] or candidates
        elif is_mountain:
            mountain_dests = ["Sa Pa", "Đà Lạt", "Hà Giang", "Mộc Châu", "Buôn Ma Thuột"]
            candidates = [t for t in candidates if t["destination"] in mountain_dests] or candidates
        elif is_south:
            south_dests = ["Cần Thơ", "An Giang"]
            candidates = [t for t in candidates if t["destination"] in south_dests] or candidates
        elif is_central:
            central_dests = ["Huế", "Hội An", "Đà Nẵng", "Quy Nhơn"]
            candidates = [t for t in candidates if t["destination"] in central_dests] or candidates
        elif is_north:
            north_dests = ["Hạ Long", "Sa Pa", "Hà Giang", "Ninh Bình", "Mộc Châu", "Cát Bà"]
            candidates = [t for t in candidates if t["destination"] in north_dests] or candidates

        # Lọc theo thời lượng
        if duration_days:
            matched_dur = [t for t in candidates if f"{duration_days} ngày" in t["duration"]]
            if matched_dur:
                candidates = matched_dur

        if is_cheap_query and candidates:
            candidates.sort(key=lambda x: x["price"])
            cheapest = candidates[0]
            return (
                f"🏷️ **Tour có chi phí tiết kiệm nhất hiện nay** là:\n"
                f"👉 **{cheapest['name']}** - Điểm đến: **{cheapest['destination']}**\n"
                f"⏱ Thời gian: {cheapest['duration']}\n"
                f"💰 Giá trọn gói: **{cheapest['price']:,.0f} VNĐ/khách**\n\n"
                f"📝 {cheapest['description']}\n\n"
                f"👉 Xem chi tiết tại: /tours/{cheapest['id']}"
            )

        if budget and candidates:
            # Phân loại theo ngân sách
            within_budget = [t for t in candidates if t["price"] <= budget]
            # Sắp xếp các tour trong ngân sách ưu tiên tour giá sát ngân sách nhất
            within_budget.sort(key=lambda x: abs(x["price"] - budget))

            # Tour chênh lệch nhẹ không quá 15%
            slightly_above = [t for t in candidates if budget < t["price"] <= budget * 1.15]
            slightly_above.sort(key=lambda x: x["price"])

            selected = within_budget[:3]
            if len(selected) < 3 and slightly_above:
                selected.extend(slightly_above[:(3 - len(selected))])

            if not selected:
                candidates.sort(key=lambda x: abs(x["price"] - budget))
                selected = candidates[:3]

            lines = [f"💡 Với mức ngân sách dự kiến khoảng **{budget:,.0f} VNĐ**, TourAI gợi ý các lựa chọn tour du lịch tối ưu nhất dành cho bạn:\n"]
            for idx, t in enumerate(selected, 1):
                price_diff = t["price"] - budget
                if price_diff <= 0:
                    status = f"✅ *(Tiết kiệm {abs(price_diff):,.0f} VNĐ)*" if price_diff < 0 else "🎯 *(Vừa vặn ngân sách)*"
                else:
                    status = f"⭐ *(Chênh lệch nhẹ +{price_diff:,.0f} VNĐ)*"

                lines.append(
                    f"{idx}. **{t['name']}** ({t['destination']})\n"
                    f"   • ⏱ Thời gian: {t['duration']}\n"
                    f"   • 💰 Giá trọn gói: **{t['price']:,.0f} VNĐ/khách** {status}\n"
                    f"   • 📝 {t['description'][:95]}...\n"
                    f"   • 👉 Xem chi tiết tại: /tours/{t['id']}\n"
                )
            lines.append("Bạn muốn tham khảo lịch trình chi tiết của tour nào trong danh sách trên?")
            return "\n".join(lines)

        return None

    def generate_response(self, question, session_id=None, force_web_search=False):
        """
        QUY TRÌNH RA QUYẾT ĐỊNH TOÀN DIỆN:
        1. Tra cứu thời tiết thời gian thực (nếu là câu hỏi thời tiết).
        2. Kế thừa ngữ cảnh hội thoại đa lượt từ ChatSessionMemory.
        3. Tư vấn theo ngân sách và sở thích bằng Smart Recommendation Engine.
        4. TF-IDF + Naive Bayes Intent Probability Distribution.
        5. So khớp Hybrid Intent-Weighted Cosine Similarity.
        6. Tra cứu Internet có định hướng cho điểm ngoài hệ thống.
        7. Trả lời từ cơ sở tri thức hoặc thuộc tính Live Database.
        """
        if not question or not question.strip():
            return "Bạn hãy nhập câu hỏi để tôi có thể tư vấn nhé."

        q_raw = question.strip()
        q_clean = correct_travel_typos(q_raw)
        q_low = q_clean.lower()

        # -------------------------------------------------------------
        # XỬ LÝ 1: TRA CỨU THỜI TIẾT THỜI GIAN THỰC (OPENWEATHERMAP)
        # -------------------------------------------------------------
        matched_local, matched_outside = self.detect_destination_context(q_clean)

        if is_weather_query(q_clean):
            display_name, query_name = extract_city_from_question(q_clean)
            if not display_name:
                if matched_local:
                    display_name = matched_local[0]["destination"]
                    query_name = remove_accents(display_name)
                elif matched_outside:
                    display_name = matched_outside[0].title()
                    query_name = remove_accents(matched_outside[0])
                else:
                    # Kế thừa điểm đến từ session trước
                    ctx = self.memory.get_context(session_id)
                    if ctx.get("active_destination"):
                        display_name = ctx["active_destination"]
                        query_name = remove_accents(display_name)

            if display_name:
                weather_data = get_weather_for_location(display_name, query_name)
                if weather_data:
                    return format_weather_response(display_name, weather_data, q_clean)

            return (
                "🌤️ Bạn đang muốn xem thông tin thời tiết ở khu vực nào? "
                "Ví dụ: 'Thời tiết TP Hồ Chí Minh', 'Thời tiết Đà Lạt hôm nay', 'Thời tiết Hà Nội'..."
            )

        # -------------------------------------------------------------
        # XỬ LÝ 2: BỘ NHỚ NGỮ CẢNH HỘI THOẠI ĐA LƯỢT (MULTI-TURN MEMORY)
        # -------------------------------------------------------------
        ctx = self.memory.get_context(session_id)
        active_tour_id = ctx.get("active_tour_id")

        # Nếu người dùng chưa nêu rõ điểm đến mới, kế thừa tour đang bàn từ lượt trước
        if not matched_local and not matched_outside and active_tour_id:
            inherited_tour = self.get_tour(active_tour_id)
            if inherited_tour:
                # Nếu câu hỏi liên quan đến lịch trình, giá cả, dịch vụ -> áp dụng tour cũ
                topic_followup = any(w in q_low for w in [
                    "lịch trình", "lich trinh", "mấy ngày", "bao lâu", "thời gian",
                    "giá", "bao nhiêu", "chi phí", "có gì", "đi đâu", "khách sạn",
                    "ăn uống", "xe đưa đón", "chuẩn bị gì", "mặc gì", "vé máy bay", "thế nào"
                ])
                if topic_followup:
                    matched_local = [inherited_tour]

        # -------------------------------------------------------------
        # XỬ LÝ 3: ĐỘNG CƠ TƯ VẤN THEO NGÂN SÁCH & SỞ THÍCH (RECOMMENDATION)
        # -------------------------------------------------------------
        budget = extract_budget(q_clean)
        recommend_triggers = [
            "tôi có", "toi co", "nên đi đâu", "nen di dau", "đi đâu", "di dau",
            "tư vấn tour", "tu van tour", "gợi ý tour", "goi y tour", "gợi ý", "goi y",
            "rẻ nhất", "re nhat", "khoảng", "tầm", "dưới", "thì đi", "thi di",
            "đi tour nào", "di tour nao", "được tour nào", "duoc tour nao",
            "chọn tour nào", "chon tour nao", "tour nào hợp", "tour nao hop",
            "có tour nào", "co tour nao", "đủ tiền", "du tien", "kinh phí", "chi phí",
            "gói tour nào", "goi tour nao", "thì đi đâu", "thi di dau", "đi đâu được", "di dau duoc"
        ]

        is_recommend_query = (
            (budget is not None and any(w in q_low for w in ["đi", "di", "tour", "đâu", "dau", "tư vấn", "tu van", "gợi ý", "goi y", "chọn", "thì", "tầm", "khoảng", "dưới", "đủ"]))
            or (any(w in q_low for w in recommend_triggers) and any(w in q_low for w in ["triệu", "trieu", "tr", "k", "tiền", "tour", "hợp", "re", "rẻ"]))
        )

        if is_recommend_query:
            rec_result = self.recommend_tours_by_criteria(q_clean)
            if rec_result:
                return rec_result

        # -------------------------------------------------------------
        # XỬ LÝ 4: TF-IDF VECTORIZATION & DEEP HYBRID INTENT PROBABILITIES
        # -------------------------------------------------------------
        processed_question = preprocess_text(q_clean)
        question_vector = self.vectorizer.transform([processed_question])

        # Phân phối xác suất tất cả các intent từ mô hình DeepHybridModel (Deep MLP + Naive Bayes)
        try:
            intent_prob_arr = self.model.predict_proba(question_vector)[0]
            intent_probs = dict(zip(self.model.classes_, intent_prob_arr))
            predicted_intent = self.model.predict(question_vector)[0]
            intent_confidence = float(max(intent_prob_arr))
        except Exception:
            predicted_intent = self.model.predict(question_vector)[0]
            intent_probs = {predicted_intent: 1.0}
            intent_confidence = 1.0

        self.last_predicted_intent = predicted_intent
        self.last_confidence = intent_confidence

        # -------------------------------------------------------------
        # XỬ LÝ 5: HYBRID INTENT-WEIGHTED COSINE SIMILARITY
        # -------------------------------------------------------------
        allowed_tour_ids = {t["id"] for t in matched_local} if matched_local else None
        best_index, cosine_score = self.compute_hybrid_cosine_similarity(
            question_vector,
            intent_probs,
            allowed_tour_ids
        )

        is_travel_topic = any(kw in q_low for kw in TRAVEL_TOPIC_KEYWORDS)
        is_explicit_web = any(kw in q_low for kw in ["tìm trên mạng", "tra google", "google", "search", "wiki", "tin tức"])

        # -------------------------------------------------------------
        # XỬ LÝ 6: ĐIỂM ĐẾN NGOÀI HỆ THỐNG HOẶC YÊU CẦU TÌM KIẾM MẠNG
        # -------------------------------------------------------------
        if force_web_search or is_explicit_web:
            web_results = search_web_for_travel(q_clean)
            if web_results:
                return format_web_response(q_raw, web_results)

        if matched_outside and not matched_local:
            out_name = matched_outside[0].title()
            out_dest_key = matched_outside[0].lower().strip()
            search_query = q_clean
            if predicted_intent in ("hoi_gia", "tour_price"):
                search_query = f"giá tour du lịch {out_name}"
            elif predicted_intent in ("hoi_lich_trinh", "tour_info"):
                search_query = f"địa điểm du lịch đẹp nổi tiếng ở {out_name}"

            web_results = search_web_for_travel(search_query)

            prefix = ""
            if any(w in q_low for w in ["tour", "giá", "chi phí", "điểm du lịch", "tham quan", "có gì", "đâu"]):
                is_abroad = any(w in q_low for w in ["nước ngoài", "quốc tế", "ngoại quốc"]) or out_dest_key in [
                    "thái lan", "nhật bản", "hàn quốc", "trung quốc", "châu âu", "châu á", "mỹ", "hoa kỳ",
                    "singapore", "malaysia", "đài loan", "bali", "úc", "pháp", "anh", "đức", "ý", "nga", "campuchia", "lào", "dubai", "hồng kông", "ấn độ"
                ]
                if is_abroad:
                    prefix = f"ℹ️ *TourAI hiện tập trung chuyên sâu 20 tuyến tour trọn gói trong nước. Dưới đây là cẩm nang du lịch và thông tin hữu ích về {out_name}:*\n\n"
                else:
                    prefix = f"ℹ️ *Hệ thống TourAI hiện chưa mở tuyến tour trọn gói tới {out_name}. Dưới đây là cẩm nang du lịch gợi ý dành cho bạn:*\n\n"

            resp = format_web_response(q_raw, web_results, destination_name=out_name)
            return prefix + resp

        # -------------------------------------------------------------
        # XỬ LÝ 6.5: TƯ VẤN CHUYÊN SÂU TOUR NỘI BỘ (RICH TOUR CONSULTATION)
        # -------------------------------------------------------------
        if matched_local:
            tour = matched_local[0]
            price_pat = r'\b(?:giá|gia|chi phí|chi phi|bao nhiêu tiền|bao tien|hết bao nhiêu|het bao nhieu|bao tiền)\b'
            schedule_pat = r'\b(?:lịch trình|lich trinh|lộ trình|lo trinh|đi đâu|di dau|tham quan gì|tham quan|kế hoạch|chương trình|chuong trinh)\b'
            duration_pat = r'\b(?:mấy ngày|may ngay|bao lâu|bao lau|thời gian|thoi gian)\b'
            faq_pat = r'\b(?:khởi hành|khoi hanh|đón ở đâu|don o dau|đón tại|don tai|trẻ em|tre em|hủy tour|huy tour|thanh toán|thanh toan|đặt cọc|dat coc|mấy giờ|may gio|được hủy|duoc huy|phương tiện|phuong tien)\b'

            has_price_word = bool(re.search(price_pat, q_low))
            has_schedule_word = bool(re.search(schedule_pat, q_low))
            has_duration_word = bool(re.search(duration_pat, q_low))
            has_faq_word = bool(re.search(faq_pat, q_low))

            is_schedule_query = has_schedule_word or (predicted_intent in ("hoi_lich_trinh", "tour_schedule") and not has_price_word and not has_faq_word)
            is_price_query = has_price_word or (predicted_intent in ("hoi_gia", "tour_price") and not has_schedule_word and not has_faq_word)
            is_general_tour_query = (
                predicted_intent in ("tour_info", "thong_tin_tour", "tim_tour", "tour_search")
                or any(k in q_low for k in ["tư vấn", "tu van", "chi tiết", "thông tin", "giới thiệu", "tour", "điểm đến"])
                or (not has_duration_word and not has_faq_word and len(q_clean.split()) <= 6)
            )

            if is_schedule_query:
                self.memory.update_context(
                    session_id,
                    active_tour_id=tour["id"],
                    active_destination=tour["destination"],
                    last_intent="hoi_lich_trinh"
                )
                return self.format_rich_tour_briefing(tour, focus="schedule")

            if is_price_query:
                self.memory.update_context(
                    session_id,
                    active_tour_id=tour["id"],
                    active_destination=tour["destination"],
                    last_intent="hoi_gia"
                )
                return self.format_rich_tour_briefing(tour, focus="price")

            if is_general_tour_query:
                self.memory.update_context(
                    session_id,
                    active_tour_id=tour["id"],
                    active_destination=tour["destination"],
                    last_intent="tour_info"
                )
                if len(matched_local) > 1 and not any(w in q_low for w in ["5 sao", "4n3đ", "3n2đ", "3 ngày", "4 ngày"]):
                    lines = [f"🏝️ Tại **{tour['destination']}**, TourAI đang có {len(matched_local)} chương trình tour trọn gói đa dạng:\n"]
                    for idx, t in enumerate(matched_local, 1):
                        lines.append(
                            f"{idx}. **{t['name']}**\n"
                            f"   • ⏱ Thời gian: {t['duration']} | 💰 Giá trọn gói: **{t['price']:,.0f} VNĐ/khách**\n"
                            f"   • 📝 {t['description']}\n"
                            f"   • 👉 Chi tiết: [{t['name']}](/tours/{t['id']})\n"
                        )
                    lines.append("Bạn muốn tham khảo lịch trình chi tiết của tour nào trong số này?")
                    return "\n".join(lines)
                else:
                    return self.format_rich_tour_briefing(tour, focus="overview")

        # -------------------------------------------------------------
        # XỬ LÝ 7: KHỚP ĐỘ TƯƠNG ĐỒNG CAO TỪ CƠ SỞ TRI THỨC NỘI BỘ (>= 0.45)
        # -------------------------------------------------------------
        if best_index >= 0 and cosine_score >= 0.45:
            # Cập nhật ngữ cảnh tour vào bộ nhớ phiên
            matched_tour_id = self.tour_ids[best_index]
            if matched_tour_id:
                tour_obj = self.get_tour(matched_tour_id)
                if tour_obj:
                    self.memory.update_context(
                        session_id,
                        active_tour_id=tour_obj["id"],
                        active_destination=tour_obj["destination"],
                        last_intent=predicted_intent
                    )
            elif matched_local:
                self.memory.update_context(
                    session_id,
                    active_tour_id=matched_local[0]["id"],
                    active_destination=matched_local[0]["destination"],
                    last_intent=predicted_intent
                )

            return self.answers[best_index]

        # -------------------------------------------------------------
        # XỬ LÝ 8: KHỚP VỪA PHẢI (0.28 <= SCORE < 0.45) KẾT HỢP LIVE TOUR DB
        # -------------------------------------------------------------
        if matched_local:
            tour = matched_local[0]
            self.memory.update_context(
                session_id,
                active_tour_id=tour["id"],
                active_destination=tour["destination"],
                last_intent=predicted_intent
            )
            return self.format_rich_tour_briefing(tour, focus="overview")

        if best_index >= 0 and cosine_score >= 0.32:
            return self.answers[best_index]

        # -------------------------------------------------------------
        # XỬ LÝ 9: GIAO TIẾP CƠ BẢN HOẶC FALLBACK TÌM KIẾM WEB
        # -------------------------------------------------------------
        if predicted_intent == "chao_hoi" or any(w in q_low for w in ["xin chào", "chào bạn", "hello", "hi", "alo", "chào"]):
            return "👋 Xin chào! Tôi là trợ lý AI của TourAI. Tôi có thể giúp bạn tìm kiếm tour du lịch, kiểm tra giá vé, xem lịch trình chi tiết và tra cứu thời tiết thời gian thực."

        if predicted_intent == "tam_biet" or any(w in q_low for w in ["tạm biệt", "cảm ơn", "bye", "hẹn gặp", "thank"]):
            return "Cảm ơn bạn đã sử dụng TourAI! Chúc bạn có những chuyến du lịch thật vui vẻ và trọn vẹn!"

        # Khi không khớp được trong tri thức nội bộ -> Tự động tìm kiếm Internet
        web_results = search_web_for_travel(q_clean)
        if web_results:
            return format_web_response(q_raw, web_results)

        return (
            "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp trong cơ sở dữ liệu cũng như trên mạng. "
            "Bạn có thể hỏi về 20 tour du lịch hiện có của TourAI (Đà Nẵng, Nha Trang, Hạ Long, Phú Quốc, Đà Lạt, Sa Pa, Quy Nhơn, Cần Thơ, Hà Giang, Ninh Bình, Huế, Hội An...), giá vé, lịch trình, thời tiết hoặc kinh nghiệm du lịch nhé!"
        )