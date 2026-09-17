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
from ai.preprocess import preprocess_text, remove_accents
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
    # Các điểm du lịch Việt Nam khác ngoài 8 tour hệ thống
    "hải phòng", "hà nội", "sài gòn", "hồ chí minh", "tphcm", "huế", "hội an",
    "côn đảo", "vũng tàu", "hà giang", "mộc châu", "ninh bình", "cát bà", "tam đảo",
    "mai châu", "ba bể", "quảng bình", "phong nha", "kẻ bàng", "bến tre", "an giang",
    "mũi né", "phan thiết", "tây bắc", "đông bắc", "đồng tháp", "bạc liêu", "cà mau",
    "bình định", "buôn ma thuột", "đắk lắk", "quảng ninh", "đồ sơn", "bạch long vĩ",
    "lý sơn", "bình ba", "nam du", "phú thọ"
]

TRAVEL_TOPIC_KEYWORDS = [
    "thời tiết", "nhiệt độ", "có mưa không", "mùa nào đẹp", "tháng mấy nên đi", "mùa bão",
    "món ăn", "ẩm thực", "đặc sản", "quán ngon", "ăn gì", "chơi gì", "uống gì", "cafe đẹp",
    "khách sạn", "homestay", "resort", "nhà nghỉ", "chỗ ở", "booking",
    "kinh nghiệm", "cẩm nang", "mẹo du lịch", "chuẩn bị gì", "mang gì", "lưu ý",
    "vé máy bay", "tàu hỏa", "xe khách", "cách đi đến", "phương tiện", "thuê xe",
    "lễ hội", "sự kiện", "bắn pháo hoa", "check in", "sống ảo", "địa điểm đẹp"
]


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
        self.model = None        # Mô hình Naive Bayes (ComplementNB / MultinomialNB)
        self.data_vectors = None # Vector TF-IDF của tập dữ liệu mẫu

        self.questions = []
        self.answers = []
        self.intents = []
        self.tour_ids = []
        self.local_tours = []

        self.memory = ChatSessionMemory()
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

        # 1. Nạp dữ liệu câu hỏi đáp và danh sách tour
        self.load_data()
        self.load_local_tours()

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
                {"id": 8, "name": "Tour Cần Thơ - Miền Tây 2 ngày 1 đêm", "destination": "Cần Thơ", "price": 2800000, "duration": "2 ngày 1 đêm", "description": "Trải nghiệm văn hóa chợ nổi Cái Răng, thưởng thức trái cây miệt vườn Nam Bộ và Bến Ninh Kiều."}
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
            "cần thơ": ["cần thơ", "can tho", "cái răng", "chợ nổi", "bến ninh kiều", "miền tây"]
        }

        matched_local = []
        for tour in self.local_tours:
            dest = tour["destination"].lower()
            kws = local_keywords_map.get(dest, [dest])
            if any(kw in q_low for kw in kws):
                matched_local.append(tour)

        matched_outside = []
        for out_dest in OUTSIDE_DESTINATIONS:
            if out_dest in q_low:
                matched_outside.append(out_dest)

        return matched_local, matched_outside

    def compute_hybrid_cosine_similarity(self, question_vector, intent_probs, allowed_tour_ids=None):
        """
        THUẬT TOÁN SO KHỚP KẾT HỢP (HYBRID INTENT-WEIGHTED COSINE SIMILARITY):
        Kết hợp trực tiếp ma trận xác suất từ Naive Bayes với độ đo Cosine TF-IDF:
        - Tính độ đo Cosine thuần túy giữa vector câu hỏi và các câu hỏi mẫu.
        - Tăng trọng số (boost) lên đến +25% cho các câu hỏi mẫu có intent trùng khớp với
          phân phối xác suất Naive Bayes.
        - Lọc ưu tiên theo allowed_tour_ids nếu có điểm đến cụ thể.
        """
        if self.data_vectors is None or self.vectorizer is None or len(self.questions) == 0:
            return -1, 0.0

        raw_similarities = cosine_similarity(question_vector, self.data_vectors)[0]

        # Áp dụng trọng số tăng cường từ Naive Bayes Intent Probabilities
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

    def recommend_tours_by_criteria(self, text):
        """
        ĐỘNG CƠ TƯ VẤN THÔNG MINH (SMART RECOMMENDATION ENGINE):
        Tự động bóc tách ngân sách, thời gian, sở thích để đề xuất danh sách tour tối ưu.
        """
        t_low = text.lower()
        budget = extract_budget(text)
        duration_days = extract_duration_days(text)

        is_cheap_query = any(w in t_low for w in ["rẻ nhất", "re nhat", "tiết kiệm", "tiet kiem", "thấp nhất"])
        is_beach = any(w in t_low for w in ["biển", "bien", "đảo", "dao", "tắm biển"])
        is_mountain = any(w in t_low for w in ["núi", "nui", "vùng cao", "săn mây", "san may", "tuyết"])

        candidates = list(self.local_tours)

        # Lọc theo sở thích
        if is_beach:
            beach_dests = ["Đà Nẵng", "Nha Trang", "Hạ Long", "Phú Quốc", "Quy Nhơn"]
            candidates = [t for t in candidates if t["destination"] in beach_dests]
        elif is_mountain:
            mountain_dests = ["Sa Pa", "Đà Lạt"]
            candidates = [t for t in candidates if t["destination"] in mountain_dests]

        # Lọc theo thời lượng
        if duration_days:
            candidates = [t for t in candidates if f"{duration_days} ngày" in t["duration"]] or candidates

        # Lọc theo ngân sách (với dung sai 10%)
        if budget:
            affordable = [t for t in candidates if t["price"] <= budget * 1.15]
            if affordable:
                candidates = affordable

        # Sắp xếp theo giá
        candidates.sort(key=lambda x: x["price"])

        if is_cheap_query and candidates:
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
            lines = [f"💡 Với mức chi phí dự kiến khoảng **{budget:,.0f} VNĐ**, TourAI xin gợi ý các tour rất phù hợp dành cho bạn:\n"]
            for idx, t in enumerate(candidates[:3], 1):
                lines.append(f"{idx}. **{t['name']}** ({t['destination']})\n   • Thời gian: {t['duration']}\n   • Giá: **{t['price']:,.0f} VNĐ**\n   • Link: /tours/{t['id']}\n")
            lines.append("Bạn muốn xem lịch trình chi tiết của tour nào trong danh sách trên?")
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
        q_low = q_raw.lower()

        # -------------------------------------------------------------
        # XỬ LÝ 1: TRA CỨU THỜI TIẾT THỜI GIAN THỰC (OPENWEATHERMAP)
        # -------------------------------------------------------------
        matched_local, matched_outside = self.detect_destination_context(q_raw)

        if is_weather_query(q_raw):
            display_name, query_name = extract_city_from_question(q_raw)
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
                    return format_weather_response(display_name, weather_data, q_raw)

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
        is_recommend_query = any(w in q_low for w in [
            "tôi có", "toi co", "nên đi đâu", "nen di dau", "tư vấn tour", "tu van tour",
            "gợi ý tour", "goi y tour", "rẻ nhất", "re nhat", "khoảng", "tầm", "dưới"
        ]) and any(w in q_low for w in ["triệu", "trieu", "tr", "k", "tiền", "tour", "hợp"])

        if is_recommend_query:
            rec_result = self.recommend_tours_by_criteria(q_raw)
            if rec_result:
                return rec_result

        # -------------------------------------------------------------
        # XỬ LÝ 4: TF-IDF VECTORIZATION & NAIVE BAYES PROBABILITIES
        # -------------------------------------------------------------
        processed_question = preprocess_text(q_raw)
        question_vector = self.vectorizer.transform([processed_question])

        # Phân phối xác suất tất cả các intent từ mô hình Naive Bayes
        try:
            intent_prob_arr = self.model.predict_proba(question_vector)[0]
            intent_probs = dict(zip(self.model.classes_, intent_prob_arr))
            predicted_intent = self.model.predict(question_vector)[0]
            intent_confidence = float(max(intent_prob_arr))
        except Exception:
            predicted_intent = self.model.predict(question_vector)[0]
            intent_probs = {predicted_intent: 1.0}
            intent_confidence = 1.0

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
            web_results = search_web_for_travel(q_raw)
            if web_results:
                return format_web_response(q_raw, web_results)

        if matched_outside and not matched_local:
            out_name = matched_outside[0].title()
            search_query = q_raw
            if predicted_intent in ("hoi_gia", "tour_price"):
                search_query = f"giá tour du lịch {out_name}"
            elif predicted_intent in ("hoi_lich_trinh", "tour_info"):
                search_query = f"địa điểm du lịch đẹp nổi tiếng ở {out_name}"

            web_results = search_web_for_travel(search_query)

            prefix = ""
            if any(w in q_low for w in ["tour", "giá", "chi phí", "điểm du lịch", "tham quan", "có gì", "đâu"]):
                if "nước ngoài" in q_low or "quốc tế" in q_low:
                    prefix = "Hiện tại hệ thống TourAI chủ yếu cung cấp các tour nội địa (Đà Nẵng, Nha Trang, Hạ Long, Phú Quốc, Đà Lạt, Sa Pa, Quy Nhơn, Cần Thơ).\n\n"
                else:
                    prefix = f"Hiện tại hệ thống TourAI chưa có tour khởi hành đến {out_name} (chúng tôi hiện có tour Đà Nẵng, Nha Trang, Hạ Long, Phú Quốc, Đà Lạt, Sa Pa, Quy Nhơn, Cần Thơ).\n\n"

            if web_results:
                return prefix + format_web_response(q_raw, web_results)
            else:
                return (
                    f"{prefix}Tôi đã tìm kiếm trên mạng về '{q_raw}' nhưng chưa có kết quả chi tiết. "
                    f"Bạn có thể tham khảo các tour hiện có (Đà Nẵng, Nha Trang, Hạ Long, Phú Quốc, Đà Lạt, Sa Pa, Quy Nhơn, Cần Thơ) nhé!"
                )

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

            if predicted_intent in ("tour_price", "hoi_gia") or any(k in q_low for k in ["giá", "chi phí", "bao nhiêu tiền", "bao tiền", "hết bao nhiêu"]):
                return f"💰 **{tour['name']}** hiện có giá trọn gói niêm yết là **{tour['price']:,.0f} VNĐ/khách**."

            elif predicted_intent in ("tour_duration", "hoi_thoi_gian") or any(k in q_low for k in ["mấy ngày", "bao lâu", "thời gian"]):
                return f"⏱ **{tour['name']}** có lịch trình gói gọn trong **{tour['duration']}**."

            elif predicted_intent in ("tour_search", "tim_tour"):
                return (
                    f"🌏 Chúng tôi có **{tour['name']}** tại **{tour['destination']}**.\n"
                    f"• Thời gian: {tour['duration']}\n"
                    f"• Giá trọn gói: **{tour['price']:,.0f} VNĐ**\n\n"
                    f"👉 Xem chi tiết tại: /tours/{tour['id']}"
                )

            elif predicted_intent in ("tour_info", "hoi_lich_trinh", "thong_tin_tour"):
                desc = tour.get("description") or "Tour trải nghiệm trọn vẹn với các điểm đến nổi tiếng, dịch vụ chất lượng."
                return (
                    f"🌏 **{tour['name']}**\n\n"
                    f"📍 Điểm đến: {tour['destination']}\n"
                    f"⏱ Thời gian: {tour['duration']}\n"
                    f"💰 Giá: **{tour['price']:,.0f} VNĐ**\n\n"
                    f"📝 {desc}\n\n"
                    f"👉 Xem chi tiết tại: /tours/{tour['id']}"
                )

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
        web_results = search_web_for_travel(q_raw)
        if web_results:
            return format_web_response(q_raw, web_results)

        return (
            "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp trong cơ sở dữ liệu cũng như trên mạng. "
            "Bạn có thể hỏi về các tour hiện có (Đà Nẵng, Nha Trang, Hạ Long, Phú Quốc, Đà Lạt, Sa Pa, Quy Nhơn, Cần Thơ), giá vé, lịch trình nhé!"
        )