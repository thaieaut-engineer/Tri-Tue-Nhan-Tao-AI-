"""
tests/test_ai_system.py
Kiểm thử toàn diện các phân hệ cốt lõi của Chatbot AI Tư Vấn Du Lịch:
1. Xử lý ngôn ngữ tự nhiên (NLP) & sửa lỗi chính tả từ vựng du lịch.
2. Trích xuất thực thể ngân sách và thời lượng (NER).
3. Động cơ điều chế và tổng hợp tri thức tìm kiếm du lịch thông minh (AI Travel Synthesis Engine).
4. Động cơ tư vấn tour chuyên sâu (Expert Tour Briefing Engine) từ lịch trình tour_schedule.
5. Bộ nhớ ngữ cảnh hội thoại đa lượt (Multi-turn Conversational Memory).
"""

import unittest
from ai.preprocess import preprocess_text, correct_travel_typos, remove_accents
from ai.chatbot import Chatbot, extract_budget, extract_duration_days
from ai.web_search import synthesize_travel_search_response, format_web_response, get_curated_destination_guide


class TestNaturalLanguageProcessing(unittest.TestCase):
    def test_travel_typo_correction(self):
        # Kiểm tra chuẩn hóa từ viết tắt và lỗi dính chữ phổ biến
        self.assertEqual(preprocess_text("tour dn gia bn"), "tour da nang gia bao nhieu")
        self.assertEqual(preprocess_text("di sapa can chuan bi gi ko"), "di sa pa can chuan bi gi khong")
        self.assertEqual(preprocess_text("co ve mb di pq ko"), "co ve may bay di phu quoc khong")

    def test_short_word_preservation(self):
        # Đảm bảo các từ ngắn không bị fuzzy matching làm biến dạng
        preprocessed = preprocess_text("Tôi muốn đi du lịch sáng mai nhận tour")
        self.assertIn("toi", preprocessed)
        self.assertIn("sang", preprocessed)
        self.assertIn("nhan", preprocessed)

    def test_budget_extraction(self):
        self.assertEqual(extract_budget("tôi có 3tr5 đi đâu"), 3500000)
        self.assertEqual(extract_budget("ngân sách tầm 4 triệu"), 4000000)
        self.assertEqual(extract_budget("khoảng 500k"), 500000)
        self.assertEqual(extract_budget("tour giá 4.500.000đ"), 4500000)

    def test_duration_extraction(self):
        self.assertEqual(extract_duration_days("tour 3 ngày 2 đêm"), 3)
        self.assertEqual(extract_duration_days("đi 4n3đ"), 4)
        self.assertEqual(extract_duration_days("khoảng 2 ngày"), 2)


class TestTravelSearchSynthesis(unittest.TestCase):
    def test_curated_destination_guide_coverage(self):
        destinations = [
            "thái lan", "nhật bản", "hàn quốc", "singapore", "trung quốc",
            "hà nội", "hồ chí minh", "quảng bình", "hải phòng", "tam đảo",
            "cà mau", "đài loan", "bali", "châu âu", "phú thọ", "lý sơn", "mai châu", "bến tre"
        ]
        for dest in destinations:
            guide = get_curated_destination_guide(dest)
            self.assertIsNotNone(guide, f"Chưa có cẩm nang cho {dest}")
            self.assertIn("title", guide)
            self.assertIn("highlights", guide)
            self.assertIn("food", guide)
            self.assertIn("best_time", guide)
            self.assertIn("budget_tips", guide)

    def test_synthesize_response_format(self):
        output = synthesize_travel_search_response(
            "Du lịch Nhật Bản có gì hay?",
            results=[{"title": "Klook Blog Nhật Bản", "url": "https://klook.com/japan", "snippet": "Nhật Bản đẹp"}],
            destination_name="nhật bản"
        )
        self.assertIn("Cẩm Nang Du Lịch Nhật Bản", output)
        self.assertIn("Ẩm thực & đặc sản trứ danh", output)
        self.assertIn("Thời điểm lý tưởng nhất", output)
        self.assertIn("Chi phí dự kiến & Thủ tục", output)
        self.assertIn("Klook Blog Nhật Bản", output)


class TestChatbotTourConsultation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = Chatbot()

    def test_schedule_consultation(self):
        resp = self.bot.generate_response("Lịch trình tour Hà Giang thế nào?")
        self.assertIn("Tour Hà Giang 3 ngày 2 đêm", resp)
        self.assertIn("Ngày 1", resp)
        self.assertIn("Ngày 2", resp)
        self.assertIn("Ngày 3", resp)
        self.assertIn("Mã Pí Lèng", resp)
        self.assertIn("Nho Quế", resp)
        self.assertIn("Ẩm thực đặc sản", resp)
        self.assertIn("Dịch vụ & tiện ích trọn gói bao gồm", resp)

    def test_price_consultation(self):
        resp = self.bot.generate_response("Giá tour Sa Pa bao nhiêu?")
        self.assertIn("BÁO GIÁ & DỊCH VỤ TRỌN GÓI", resp)
        self.assertIn("4,100,000", resp)
        self.assertIn("Giá tour đã bao gồm trọn gói", resp)

    def test_multi_tour_destination_selection(self):
        resp = self.bot.generate_response("Tour Phú Quốc")
        self.assertIn("Tour Phú Quốc 3 ngày 2 đêm", resp)
        self.assertIn("Tour Phú Quốc Nghỉ Dưỡng 5 Sao 4 ngày 3 đêm", resp)
        self.assertIn("/tours/4", resp)
        self.assertIn("/tours/20", resp)

    def test_multiturn_conversational_memory(self):
        session_id = "test_unit_session_001"
        r1 = self.bot.generate_response("Tư vấn tour Nha Trang", session_id=session_id)
        self.assertIn("Nha Trang", r1)

        # Hỏi câu tiếp nối không nhắc lại chữ "Nha Trang"
        r2 = self.bot.generate_response("Lịch trình chi tiết thế nào?", session_id=session_id)
        self.assertIn("Nha Trang", r2)
        self.assertIn("Ngày 1", r2)
        self.assertIn("VinWonders", r2)

    def test_monthly_travel_consultation(self):
        resp = self.bot.generate_response("địa điểm du lịch phù hợp cho tháng 10")
        self.assertIn("THÁNG 10", resp)
        self.assertIn("Hà Giang", resp)
        self.assertIn("tam giác mạch", resp.lower())
        self.assertIn("/tours/9", resp)

    def test_seasonal_travel_consultation(self):
        resp = self.bot.generate_response("mùa hè nên đi đâu tránh nóng")
        self.assertIn("MÙA HÈ", resp)
        self.assertIn("Đà Nẵng", resp)
        self.assertIn("/tours/1", resp)

    def test_audience_travel_consultation(self):
        resp = self.bot.generate_response("gia đình có con nhỏ nên đi du lịch ở đâu")
        self.assertIn("TRẺ NHỎ", resp)
        self.assertIn("Đà Nẵng", resp)
        self.assertIn("/tours/1", resp)

    def test_thematic_travel_consultation(self):
        resp = self.bot.generate_response("đi săn mây ở đâu đẹp nhất")
        self.assertIn("SĂN BIỂN MÂY", resp)
        self.assertIn("Sa Pa", resp)
        self.assertIn("/tours/6", resp)

    def test_hotel_theft_incident_consultation(self):
        # Kiểm tra xử lý sự cố mất đồ tại khách sạn (Issue 1)
        resp = self.bot.generate_response("đang hỏi trong trường hợp nếu t bị ăn cắp mất đồ ở khách sạn thì giải quyết như nào?")
        self.assertIn("MẤT ĐỒ / MẤT CẮP TẠI KHÁCH SẠN", resp)
        self.assertIn("Giữ nguyên hiện trường", resp)
        self.assertIn("Lễ tân", resp)
        self.assertIn("Biên Bản Ghi Nhận Sự Việc", resp)
        self.assertIn("Công an", resp)
        self.assertIn("50.000.000", resp)
        # Đảm bảo không bắt nhầm sang chính sách hoàn hủy bão lũ thiên tai
        self.assertNotIn("bão lũ", resp)
        self.assertNotIn("hoàn lại 100% chi phí", resp)

    def test_general_theft_incident_consultation(self):
        # Kiểm tra tư vấn tình huống mất cắp khi đi du lịch (Issue 2)
        resp = self.bot.generate_response("nếu tôi bị ăn cắp thì sao")
        self.assertIn("MẤT CẮP", resp)
        self.assertIn("Khóa khẩn cấp thẻ ngân hàng", resp)
        self.assertIn("Công an", resp)
        self.assertIn("Bảo hiểm", resp)
        # Đảm bảo không bị web search tìm nhầm sang hội chứng tâm thần
        self.assertNotIn("tâm thần", resp.lower())
        self.assertNotIn("hội chứng ăn cắp", resp.lower())

    def test_tour_guide_language_french_group(self):
        # Kiểm tra tư vấn nghiệp vụ HDV ngoại ngữ & đoàn khách Pháp (Issue 3)
        resp = self.bot.generate_response("hướng dẫn viên nói được tiếng anh ko, đoàn tôi là đoàn nước pháp")
        self.assertIn("HƯỚNG DẪN VIÊN", resp)
        self.assertIn("Tiếng Anh", resp)
        self.assertIn("tiếng Pháp", resp)
        self.assertIn("Thẻ Hướng dẫn viên Quốc tế", resp)
        # Đảm bảo không bắt nhầm nước Pháp thành tour du lịch Châu Âu / Tháp Eiffel
        self.assertNotIn("Tháp Eiffel", resp)
        self.assertNotIn("Bảo tàng Louvre", resp)

    def test_pure_consultative_tone_no_sales_pitch(self):
        # Kiểm tra chatbot giữ đúng vai trò tư vấn, không chèo kéo bán tour
        resp = self.bot.generate_response("Du lịch Thái Lan có gì hay?")
        self.assertNotIn("TourAI hiện tập trung chuyên sâu phục vụ 20 tuyến tour trọn gói", resp)
        self.assertIn("Cẩm Nang Du Lịch Thái Lan", resp)


if __name__ == "__main__":
    unittest.main()


