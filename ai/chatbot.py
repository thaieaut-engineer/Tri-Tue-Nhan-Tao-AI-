"""
ai/chatbot.py
Module Chatbot xử lý ngôn ngữ tự nhiên ứng dụng kết hợp 3 kỹ thuật cốt lõi:
 1. TF-IDF (TfidfVectorizer): Trích xuất đặc trưng từ câu hỏi và biểu diễn dưới dạng không gian vector.
 2. Naive Bayes (MultinomialNB): Mô hình học máy phân loại ý định (Intent Classification).
 3. Cosine Similarity (Độ đo tương đồng cosine): So khớp độ tương đồng ngữ nghĩa giữa câu hỏi người dùng và tập dữ liệu.
Khi câu hỏi ngoài phạm vi cơ sở dữ liệu nội bộ, chatbot dùng intent dự đoán bởi Naive Bayes để tìm kiếm trên Internet.
"""

import os
import json
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


# Danh sách các điểm đến ngoài hệ thống hoặc du lịch nước ngoài
OUTSIDE_DESTINATIONS = [
    # Nước ngoài / Quốc tế
    "nước ngoài", "quốc tế", "ngoại quốc", "thái lan", "nhật bản", "hàn quốc", "trung quốc",
    "châu âu", "châu á", "mỹ", "hoa kỳ", "singapore", "malaysia", "đài loan", "bali", "úc",
    "pháp", "anh", "đức", "ý", "nga", "campuchia", "lào", "dubai", "hồng kông", "ấn độ",
    # Các tỉnh thành / địa danh du lịch Việt Nam khác ngoài 8 tour trong DB
    "hải phòng", "hà nội", "sài gòn", "hồ chí minh", "tphcm", "huế", "hội an",
    "côn đảo", "vũng tàu", "hà giang", "mộc châu", "ninh bình", "cát bà", "tam đảo",
    "mai châu", "ba bể", "quảng bình", "phong nha", "kẻ bàng", "bến tre", "an giang",
    "mũi né", "phan thiết", "tây bắc", "đông bắc", "đồng tháp", "bạc liêu", "cà mau",
    "bình định", "buôn ma thuột", "đắk lắk", "quảng ninh", "đồ sơn", "bạch long vĩ",
    "lý sơn", "bình ba", "nam du", "phú thọ"
]

# Các chủ đề du lịch mở rộng
TRAVEL_TOPIC_KEYWORDS = [
    "thời tiết", "nhiệt độ", "có mưa không", "mùa nào đẹp", "tháng mấy nên đi", "mùa bão",
    "món ăn", "ẩm thực", "đặc sản", "quán ngon", "ăn gì", "chơi gì", "uống gì", "cafe đẹp",
    "khách sạn", "homestay", "resort", "nhà nghỉ", "chỗ ở", "booking",
    "kinh nghiệm", "cẩm nang", "mẹo du lịch", "chuẩn bị gì", "mang gì", "lưu ý",
    "vé máy bay", "tàu hỏa", "xe khách", "cách đi đến", "phương tiện", "thuê xe",
    "lễ hội", "sự kiện", "bắn pháo hoa", "check in", "sống ảo", "địa điểm đẹp"
]


class Chatbot:

    def __init__(self):
        self.vectorizer = None   # Mô hình TF-IDF
        self.model = None        # Mô hình Naive Bayes (MultinomialNB)
        self.data_vectors = None # Vector TF-IDF của tập dữ liệu mẫu

        self.questions = []
        self.answers = []
        self.intents = []
        self.tour_ids = []
        self.local_tours = []

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
        """
        Lấy danh sách các tour và điểm đến trong hệ thống để đối chiếu thực thể.
        """
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
        """
        Đọc dữ liệu câu hỏi mẫu, câu trả lời và ý định (intent) từ MySQL hoặc fallback file JSON.
        """
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
        """
        Nhận diện thực thể điểm đến (Entity Recognition) để phân biệt:
        - Điểm đến nội bộ (matched_local)
        - Điểm đến ngoài hệ thống hoặc du lịch nước ngoài (matched_outside)
        """
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

    def compute_cosine_similarity(self, question_vector, allowed_tour_ids=None):
        """
        ÁP DỤNG COSINE SIMILARITY:
        Tính độ tương đồng Cosine giữa vector TF-IDF của câu hỏi người dùng
        với toàn bộ ma trận vector TF-IDF của tập câu hỏi mẫu.
        Nếu allowed_tour_ids được chỉ định, ưu tiên các câu hỏi thuộc tour tương ứng
        hoặc câu hỏi chung (tour_id is None).
        """
        if self.data_vectors is None or self.vectorizer is None or len(self.questions) == 0:
            return -1, 0.0

        # Tính toán ma trận độ tương đồng Cosine
        similarities = cosine_similarity(question_vector, self.data_vectors)[0]

        if allowed_tour_ids:
            # Lọc các index phù hợp (cùng tour_id hoặc tour_id is None)
            valid_indices = [
                i for i, tid in enumerate(self.tour_ids)
                if tid is None or tid in allowed_tour_ids
            ]
            if valid_indices:
                filtered_sims = similarities[valid_indices]
                best_sub_idx = int(filtered_sims.argmax())
                best_index = valid_indices[best_sub_idx]
                best_score = float(similarities[best_index])
                return best_index, best_score

        best_index = int(similarities.argmax())
        best_score = float(similarities[best_index])

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

    def generate_response(self, question, force_web_search=False):
        """
        QUY TRÌNH XỬ LÝ KẾT HỢP: TF-IDF + NAIVE BAYES + COSINE SIMILARITY
        1. Tiền xử lý văn bản (loại bỏ dấu, chuẩn hóa ký tự).
        2. TF-IDF Vectorizer: Chuyển đổi câu hỏi thành vector đặc trưng.
        3. Naive Bayes (MultinomialNB): Dự đoán phân loại Ý định (Intent Classification).
        4. Cosine Similarity: Đo lường độ tương đồng ngữ nghĩa giữa vector câu hỏi và cơ sở tri thức.
        5. Ra quyết định:
           - Khớp chính xác nội bộ (Cosine Similarity >= 0.50) -> Trả về câu trả lời chi tiết, chuẩn xác từ cơ sở tri thức.
           - Khớp tương đối theo tour nội bộ (0.30 <= Cosine < 0.50) -> Tổng hợp trực tiếp từ bảng dữ liệu tour theo Intent.
           - Điểm đến ngoài hệ thống hoặc độ tương đồng thấp -> Tận dụng Naive Bayes Intent kết hợp tra cứu Internet có định hướng!
        """
        if not question or not question.strip():
            return "Bạn hãy nhập câu hỏi để tôi có thể tư vấn nhé."

        q_raw = question.strip()
        q_low = q_raw.lower()

        # Nhận diện ngữ cảnh điểm đến
        matched_local, matched_outside = self.detect_destination_context(q_raw)
        is_travel_topic = any(kw in q_low for kw in TRAVEL_TOPIC_KEYWORDS)
        is_explicit_web = any(kw in q_low for kw in ["tìm trên mạng", "tra google", "google", "search", "wiki", "tin tức"])

        # -------------------------------------------------------------
        # XỬ LÝ ĐẶC THÙ: TRA CỨU THỜI TIẾT (OPENWEATHERMAP / OPEN-METEO API)
        # -------------------------------------------------------------
        if is_weather_query(q_raw):
            display_name, query_name = extract_city_from_question(q_raw)
            if not display_name:
                if matched_local:
                    display_name = matched_local[0]["destination"]
                    query_name = remove_accents(display_name)
                elif matched_outside:
                    display_name = matched_outside[0].title()
                    query_name = remove_accents(matched_outside[0])

            if display_name:
                weather_data = get_weather_for_location(display_name, query_name)
                if weather_data:
                    return format_weather_response(display_name, weather_data, q_raw)

            return (
                "🌤️ Bạn đang muốn xem thông tin thời tiết ở khu vực nào? "
                "Ví dụ: 'Thời tiết TP Hồ Chí Minh', 'Thời tiết Đà Lạt hôm nay', 'Thời tiết Hà Nội'..."
            )

        # -------------------------------------------------------------
        # BƯỚC 1: TF-IDF VECTORIZATION
        # -------------------------------------------------------------
        processed_question = preprocess_text(q_raw)
        question_vector = self.vectorizer.transform([processed_question])

        # -------------------------------------------------------------
        # BƯỚC 2: NAIVE BAYES INTENT CLASSIFICATION
        # Dự đoán phân loại ý định người dùng
        # -------------------------------------------------------------
        predicted_intent = self.model.predict(question_vector)[0]
        try:
            intent_probs = self.model.predict_proba(question_vector)[0]
            intent_confidence = float(max(intent_probs))
        except Exception:
            intent_confidence = 1.0

        # -------------------------------------------------------------
        # BƯỚC 3: COSINE SIMILARITY MATCHING
        # Đo độ tương đồng ngữ nghĩa, ưu tiên các câu hỏi liên quan đến điểm đến nếu có
        # -------------------------------------------------------------
        allowed_tour_ids = {t["id"] for t in matched_local} if matched_local else None
        best_index, cosine_score = self.compute_cosine_similarity(question_vector, allowed_tour_ids)

        # -------------------------------------------------------------
        # BƯỚC 4: XỬ LÝ KHI NGƯỜI DÙNG YÊU CẦU TRA CỨU MẠNG HOẶC HỎI ĐIỂM NGOÀI HỆ THỐNG
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
        # BƯỚC 5: KHỚP CHÍNH XÁC NỘI BỘ VỚI COSINE SIMILARITY >= 0.50
        # Ưu tiên các câu trả lời chuyên sâu, đã biên soạn hoàn chỉnh
        # -------------------------------------------------------------
        if best_index >= 0 and cosine_score >= 0.50:
            return self.answers[best_index]

        # -------------------------------------------------------------
        # BƯỚC 6: KHỚP VỪA PHẢI (0.30 <= COSINE < 0.50) VỚI TOUR NỘI BỘ
        # Dùng Naive Bayes Predicted Intent kết hợp thuộc tính từ bảng Tour
        # -------------------------------------------------------------
        if matched_local:
            tour = matched_local[0]
            if predicted_intent in ("tour_price", "hoi_gia") or any(k in q_low for k in ["giá", "chi phí", "bao nhiêu tiền", "bao tiền", "hết bao nhiêu"]):
                return f"💰 {tour['name']} có giá {tour['price']:,.0f} VNĐ."

            elif predicted_intent in ("tour_duration", "hoi_thoi_gian") or any(k in q_low for k in ["mấy ngày", "bao lâu", "thời gian"]):
                return f"⏱ {tour['name']} có thời gian {tour['duration']}."

            elif predicted_intent in ("tour_search", "tim_tour"):
                return (
                    f"🌏 Chúng tôi có {tour['name']} tại {tour['destination']}.\n"
                    f"Thời gian: {tour['duration']} | Giá: {tour['price']:,.0f} VNĐ.\n"
                    f"Bạn có thể xem chi tiết tại: /tours/{tour['id']}"
                )

            elif predicted_intent in ("tour_info", "hoi_lich_trinh", "thong_tin_tour"):
                desc = tour.get("description") or "Tour trải nghiệm trọn vẹn với các điểm đến nổi tiếng, dịch vụ chất lượng."
                return (
                    f"🌏 {tour['name']}\n\n"
                    f"📍 Điểm đến: {tour['destination']}\n"
                    f"⏱ Thời gian: {tour['duration']}\n"
                    f"💰 Giá: {tour['price']:,.0f} VNĐ\n\n"
                    f"📝 {desc}\n\n"
                    f"👉 Xem chi tiết tại: /tours/{tour['id']}"
                )

        if best_index >= 0 and cosine_score >= 0.35:
            return self.answers[best_index]

        # -------------------------------------------------------------
        # BƯỚC 7: XỬ LÝ Ý ĐỊNH GIAO TIẾP CƠ BẢN HOẶC TẬN DỤNG TÌM KIẾM WEB
        # -------------------------------------------------------------
        if predicted_intent == "chao_hoi" or any(w in q_low for w in ["xin chào", "chào bạn", "hello", "hi", "alo", "chào"]):
            return "👋 Xin chào! Tôi là trợ lý AI của TourAI. Tôi có thể giúp bạn tìm kiếm tour du lịch, kiểm tra giá vé, xem lịch trình chi tiết và tra cứu thời tiết thời gian thực."

        if predicted_intent == "tam_biet" or any(w in q_low for w in ["tạm biệt", "cảm ơn", "bye", "hẹn gặp", "thank"]):
            return "Cảm ơn bạn đã sử dụng TourAI! Chúc bạn có những chuyến du lịch thật vui vẻ và trọn vẹn!"

        # Nếu là câu hỏi về chủ đề du lịch nói chung hoặc câu hỏi mở rộng
        if is_travel_topic or is_explicit_web:
            web_results = search_web_for_travel(q_raw)
            if web_results:
                return format_web_response(q_raw, web_results)

        # Fallback tìm kiếm web cho các câu hỏi chưa rõ ràng
        web_results = search_web_for_travel(q_raw)
        if web_results:
            return format_web_response(q_raw, web_results)

        return (
            "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp trong cơ sở dữ liệu cũng như trên mạng. "
            "Bạn có thể hỏi về các tour hiện có (Đà Nẵng, Nha Trang, Hạ Long, Phú Quốc, Đà Lạt, Sa Pa, Quy Nhơn, Cần Thơ), giá vé, lịch trình nhé!"
        )