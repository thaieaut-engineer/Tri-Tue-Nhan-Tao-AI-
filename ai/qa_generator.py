"""
ai/qa_generator.py
HỆ THỐNG SINH CÂU HỎI DU LỊCH TỰ ĐỘNG & ĐƯỜNG ỐNG THU THẬP DỮ LIỆU HUẤN LUYỆN
(AUTOMATED TRAVEL QUESTION GENERATOR & CONTINUAL RETRAINING PIPELINE)

Các tính năng chính:
1. TravelQuestionGenerator:
   - Tự động sinh hàng trăm mẫu câu hỏi đa dạng ngữ cảnh và tự nhiên.
   - 7 chuyên mục: Khách sạn & Resort, Quy mô đoàn & Báo giá nhóm, Ngân sách,
     Lịch trình & Hoạt động, Mùa vụ & Thời tiết & Cẩm nang, Dịch vụ & Chính sách, Điểm đến nước ngoài.
   - Kết hợp linh hoạt tiền tố, hậu tố, slot-filling (địa danh, ngân sách, số khách, số sao, mùa vụ).
2. DataHarvestingTrainingPipeline:
   - Gửi câu hỏi qua Chatbot để thu thập câu trả lời tư vấn sâu và bóc tách Intent / Confidence.
   - Bộ lọc kiểm duyệt chất lượng (Quality Validation & Deduplication).
   - Tự động nạp dữ liệu vào MySQL (qa_data) và data/sample_qa.json.
   - Kích hoạt quy trình tái huấn luyện mô hình Deep Learning PyTorch (PyTorchDeepIntentNet)
     cùng việc cập nhật không gian vector nhúng ngữ nghĩa tiềm ẩn 64-D (64-D Latent Semantic Embeddings).
   - Tự động nạp nóng (Hot-reload) Chatbot để kích hoạt tri thức mới tức thì.
"""

import os
import sys
import time
import json
import random
from datetime import datetime

from ai.preprocess import preprocess_text, remove_accents
from database.db import get_connection
from models.qa_data import create_qa, get_all_qa, count_qa


DESTINATIONS = [
    "Đà Nẵng", "Nha Trang", "Phú Quốc", "Đà Lạt", "Sa Pa", "Hạ Long",
    "Ninh Bình", "Hà Giang", "Quy Nhơn", "Huế", "Hội An", "Cát Bà",
    "Buôn Ma Thuột", "Mộc Châu", "Cần Thơ", "Phan Thiết", "Vũng Tàu",
    "Hà Nội", "TP Hồ Chí Minh"
]

OVERSEAS_DESTINATIONS = [
    "Thái Lan", "Nhật Bản", "Hàn Quốc", "Singapore", "Pháp", "Đài Loan", "Bali"
]

GROUP_SIZES = [3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30]

BUDGETS = [2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 15]

DURATIONS = [
    "2 ngày 1 đêm", "3 ngày 2 đêm", "4 ngày 3 đêm", "5 ngày 4 đêm", "1 ngày"
]

MONTHS = list(range(1, 13))

SEASONS = ["xuân", "hạ", "thu", "đông"]

HOTEL_STARS = ["3 sao", "4 sao", "5 sao"]

LANGUAGES = ["tiếng Anh", "tiếng Trung", "tiếng Hàn", "tiếng Nhật", "tiếng Pháp"]

PREFIXES = [
    "Cho mình hỏi ", "Ad ơi ", "Bạn ơi ", "Cho tớ hỏi ", "Xin hỏi ",
    "TourAI ơi ", "Tư vấn giúp em ", "Cho hỏi thăm ", "Dạ cho em hỏi ", ""
]

SUFFIXES = [
    " vậy ạ?", " không bot?", " được không?", " nhé!", " với ạ.",
    " thế bạn?", " ạ?", " chi tiết giúp mình.", "?"
]


TEMPLATES = {
    "hotel": {
        "expected_intent": "hoi_dich_vu",
        "patterns": [
            "khách sạn ở {dest} mấy sao",
            "tour {dest} ở khách sạn nào vậy",
            "bên bạn cho mình ở khách sạn nào tại {dest}",
            "khách sạn tại {dest} có tiêu chuẩn 5 sao không",
            "nêu tên khách sạn đối tác ở {dest}",
            "danh sách khách sạn 4 sao tại {dest} của tour",
            "ở {dest} có những resort nào hợp tác",
            "tiêu chuẩn khách sạn đi tour {dest} là mấy sao",
            "tour {dest} có phòng riêng cho 2 người không",
            "khách sạn ở {dest} có kèm buffet sáng không"
        ]
    },
    "group": {
        "expected_intent": "tu_van_dat_tour",
        "patterns": [
            "đoàn mình {group_size} người đi {dest} chi phí khoảng bao nhiêu",
            "nhóm {group_size} khách du lịch {dest} {duration} có ưu đãi gì không",
            "giá tour {dest} cho đoàn {group_size} người lớn",
            "nhà tôi {group_size} người muốn đi {dest} tính tổng cộng hết bao tiền",
            "đoàn {group_size} người đi {dest} có bố trí xe riêng không",
            "chi phí trọn gói đi {dest} cho nhóm {group_size} người",
            "tư vấn tour {dest} cho đoàn công ty {group_size} người"
        ]
    },
    "budget": {
        "expected_intent": "tu_van_dat_tour",
        "patterns": [
            "tôi có {budget} triệu thì nên đi du lịch ở đâu",
            "với kinh phí {budget}tr có đi được {dest} không",
            "gợi ý tour {dest} tầm {budget} triệu",
            "có tour {dest} nào giá dưới {budget} triệu không",
            "ngân sách khoảng {budget} triệu đi tour nào hợp lý nhất",
            "mình có tầm {budget}tr muốn đi chơi đâu đó tư vấn giúp mình"
        ]
    },
    "schedule": {
        "expected_intent": "hoi_lich_trinh",
        "patterns": [
            "lịch trình tour {dest} {duration} chi tiết gồm những gì",
            "ngày 1 và ngày 2 ở {dest} đi tham quan những điểm nào",
            "tour {dest} có đi cáp treo hoặc lặn ngắm san hô không",
            "thời gian khám phá {dest} mấy ngày mấy đêm là hợp lý",
            "chi tiết các hoạt động trong tour {dest}",
            "lịch trình đi {dest} xuất phát từ mấy giờ"
        ]
    },
    "weather": {
        "expected_intent": "kinh_nghiem_du_lich",
        "patterns": [
            "tháng {month} nên đi du lịch ở đâu đẹp nhất",
            "mùa {season} ở {dest} thời tiết thế nào, có mưa không",
            "đi du lịch {dest} nên chuẩn bị trang phục và đồ đạc gì",
            "đến {dest} có đặc sản gì ngon nên thử",
            "thời điểm lý tưởng nhất trong năm để đi {dest} là khi nào",
            "tháng {month} đi {dest} có tắm biển được không"
        ]
    },
    "policy": {
        "expected_intent": "hoi_dich_vu",
        "patterns": [
            "giá tour {dest} đã bao gồm vé máy bay và ăn uống chưa",
            "bên bạn có hướng dẫn viên {language} cho đoàn không",
            "chính sách hủy đổi tour hoặc bảo hiểm du lịch bên mình thế nào",
            "trẻ em đi tour {dest} được tính giá vé như thế nào",
            "tiêu chuẩn bữa ăn trong tour {dest} ra sao",
            "bảo hiểm du lịch của TourAI hạn mức bồi thường là bao nhiêu"
        ]
    },
    "overseas": {
        "expected_intent": "tu_van_dat_tour",
        "patterns": [
            "bên bạn có tour đi {overseas_dest} không",
            "tư vấn du lịch {overseas_dest} mùa này có gì đẹp",
            "chi phí đi du lịch {overseas_dest} khoảng bao nhiêu tiền",
            "đi {overseas_dest} cần chuẩn bị thủ tục visa và giấy tờ gì không"
        ]
    }
}


class TravelQuestionGenerator:
    """
    TRÌNH SINH CÂU HỎI DU LỊCH TỰ ĐỘNG (PARAMETRIC QUESTION GENERATOR):
    Tạo ra tập dữ liệu câu hỏi phong phú, tự nhiên phục vụ kiểm thử và học liên tục.
    """

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.categories = list(TEMPLATES.keys())

    def get_supported_categories(self):
        return list(self.categories)

    def generate_single(self, category=None):
        """Sinh một câu hỏi ngẫu nhiên thuộc danh mục chỉ định hoặc ngẫu nhiên."""
        if not category or category not in TEMPLATES:
            category = random.choice(self.categories)

        cat_config = TEMPLATES[category]
        pattern = random.choice(cat_config["patterns"])
        prefix = random.choice(PREFIXES)
        suffix = random.choice(SUFFIXES)

        dest = random.choice(DESTINATIONS)
        overseas_dest = random.choice(OVERSEAS_DESTINATIONS)
        group_size = random.choice(GROUP_SIZES)
        budget = random.choice(BUDGETS)
        duration = random.choice(DURATIONS)
        month = random.choice(MONTHS)
        season = random.choice(SEASONS)
        star = random.choice(HOTEL_STARS)
        language = random.choice(LANGUAGES)

        budget_str = f"{budget:g}"

        formatted_core = pattern.format(
            dest=dest,
            overseas_dest=overseas_dest,
            group_size=group_size,
            budget=budget_str,
            duration=duration,
            month=month,
            season=season,
            star=star,
            language=language
        )

        # Ráp prefix và suffix
        question_text = f"{prefix}{formatted_core}{suffix}".strip()
        # Chuẩn hóa khoảng trắng
        question_text = " ".join(question_text.split())
        # Viết hoa chữ cái đầu tiên
        if question_text:
            question_text = question_text[0].upper() + question_text[1:]

        metadata = {
            "category": category,
            "destination": dest if "{dest}" in pattern else (overseas_dest if "{overseas_dest}" in pattern else None),
            "group_size": group_size if "{group_size}" in pattern else None,
            "budget": budget if "{budget}" in pattern else None,
            "duration": duration if "{duration}" in pattern else None,
            "month": month if "{month}" in pattern else None,
            "season": season if "{season}" in pattern else None,
            "star": star if "{star}" in pattern else None,
            "language": language if "{language}" in pattern else None
        }

        return {
            "question": question_text,
            "category": category,
            "expected_intent": cat_config["expected_intent"],
            "metadata": metadata
        }

    def generate_batch(self, count=30, categories=None, ensure_unique=True, exclude_questions=None):
        """
        Sinh một mẻ gồm `count` câu hỏi đa dạng, đảm bảo không trùng lặp nội dung.
        """
        if categories:
            valid_cats = [c for c in categories if c in TEMPLATES]
            if not valid_cats:
                valid_cats = self.categories
        else:
            valid_cats = self.categories

        exclude_set = set()
        if exclude_questions:
            for q in exclude_questions:
                exclude_set.add(preprocess_text(q))

        generated_items = []
        seen_in_batch = set()
        max_attempts = count * 15
        attempts = 0

        while len(generated_items) < count and attempts < max_attempts:
            attempts += 1
            cat = random.choice(valid_cats)
            item = self.generate_single(category=cat)
            norm = preprocess_text(item["question"])

            if ensure_unique:
                if norm in seen_in_batch or norm in exclude_set:
                    continue

            seen_in_batch.add(norm)
            generated_items.append(item)

        return generated_items


class DataHarvestingTrainingPipeline:
    """
    ĐƯỜNG ỐNG THU THẬP DỮ LIỆU & HUẤN LUYỆN LẠI HỌC SÂU LIÊN TỤC:
    1. Tiếp nhận các câu hỏi sinh tự động.
    2. Gọi Chatbot để sản sinh câu trả lời tư vấn sâu và thu thập metadata.
    3. Thẩm định chất lượng dữ liệu Q&A thu hoạch được.
    4. Nạp vào MySQL qa_data & data/sample_qa.json.
    5. Tái huấn luyện mạng nơ-ron sâu PyTorch (PyTorchDeepIntentNet) và cập nhật 64-D Latent Semantic Embeddings.
    6. Hot-reload Chatbot.
    """

    def __init__(self, chatbot_instance=None):
        if chatbot_instance is not None:
            self.bot = chatbot_instance
        else:
            from ai.chatbot import Chatbot
            self.bot = Chatbot()

    def get_existing_questions_set(self):
        """Tập hợp tất cả các câu hỏi đã có sẵn trong hệ thống để tránh trùng lặp."""
        existing_set = set()
        # 1. Từ sample_qa.json
        json_path = os.path.abspath("data/sample_qa.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for it in data:
                        existing_set.add(preprocess_text(it.get("question", "")))
            except Exception as e:
                print("Lỗi đọc sample_qa.json:", e)

        # 2. Từ MySQL qa_data
        try:
            qa_rows = get_all_qa()
            for r in qa_rows:
                existing_set.add(preprocess_text(r.get("question", "")))
        except Exception:
            pass

        return existing_set

    def harvest_answers(self, questions_data, delay_ms=0, verbose=False):
        """
        Gửi danh sách câu hỏi qua Chatbot, thu hoạch câu trả lời và đánh giá chất lượng.
        """
        harvested = []
        for idx, item in enumerate(questions_data, 1):
            q_text = item["question"]
            start_t = time.time()

            try:
                # Gọi Chatbot xử lý câu hỏi
                ans_text = self.bot.generate_response(q_text)
                latency = round((time.time() - start_t) * 1000, 2)

                # Thu thập intent và confidence
                detected_intent = getattr(self.bot, "last_predicted_intent", None) or item.get("expected_intent", "tu_van_dat_tour")
                confidence = getattr(self.bot, "last_confidence", 0.95)
                if confidence is None:
                    confidence = 0.95

                # Kiểm tra điểm đến khớp
                matched_local, _ = self.bot.detect_destination_context(q_text)
                tour_id = matched_local[0]["id"] if matched_local else None

                # Thẩm định chất lượng câu trả lời
                is_valid = True
                invalid_reasons = []

                if not ans_text or len(ans_text.strip()) < 30:
                    is_valid = False
                    invalid_reasons.append("Câu trả lời quá ngắn (< 30 ký tự)")

                error_phrases = ["tôi chưa hiểu rõ", "lỗi hệ thống", "không có dữ liệu"]
                if any(p in ans_text.lower() for p in error_phrases):
                    is_valid = False
                    invalid_reasons.append("Chứa thông báo lỗi fallback")

                record = {
                    "question": q_text,
                    "answer": ans_text,
                    "intent": detected_intent,
                    "expected_intent": item.get("expected_intent"),
                    "confidence": round(float(confidence), 4),
                    "tour_id": tour_id,
                    "category": item.get("category", "general"),
                    "metadata": item.get("metadata", {}),
                    "latency_ms": latency,
                    "is_valid": is_valid,
                    "invalid_reasons": invalid_reasons
                }
                harvested.append(record)

                if verbose:
                    status_icon = "✅" if is_valid else "⚠️"
                    print(f"[{idx}/{len(questions_data)}] {status_icon} [{record['intent']}|{record['confidence']*100:.1f}%] {q_text[:45]}... ({latency:.0f}ms)")

            except Exception as e:
                harvested.append({
                    "question": q_text,
                    "answer": "",
                    "intent": item.get("expected_intent", "tu_van_dat_tour"),
                    "expected_intent": item.get("expected_intent"),
                    "confidence": 0.0,
                    "tour_id": None,
                    "category": item.get("category", "general"),
                    "metadata": item.get("metadata", {}),
                    "latency_ms": round((time.time() - start_t) * 1000, 2),
                    "is_valid": False,
                    "invalid_reasons": [f"Lỗi ngoại lệ: {str(e)}"]
                })

            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

        return harvested

    def ingest_and_sync(self, harvested_items, persist_to_db=True, persist_to_json=True, json_path=None):
        """
        Nạp các bản ghi Q&A hợp lệ vào MySQL qa_data và đồng bộ data/sample_qa.json.
        """
        valid_items = [it for it in harvested_items if it.get("is_valid", False)]
        if not valid_items:
            return {
                "success": True,
                "added_count": 0,
                "duplicates_skipped": 0,
                "message": "Không có bản ghi Q&A hợp lệ nào cần lưu."
            }

        existing_set = self.get_existing_questions_set()
        to_add = []
        duplicates_skipped = 0

        for it in valid_items:
            q_norm = preprocess_text(it["question"])
            if q_norm in existing_set:
                duplicates_skipped += 1
                continue
            existing_set.add(q_norm)
            to_add.append(it)

        if not to_add:
            return {
                "success": True,
                "added_count": 0,
                "duplicates_skipped": duplicates_skipped,
                "message": f"Tất cả {len(valid_items)} câu hỏi đều đã tồn tại trong cơ sở dữ liệu."
            }

        added_count = 0
        added_records = []

        # 1. Lưu vào MySQL qa_data
        if persist_to_db:
            for it in to_add:
                try:
                    new_id = create_qa(
                        question=it["question"],
                        answer=it["answer"],
                        intent=it["intent"],
                        tour_id=it.get("tour_id")
                    )
                    if new_id:
                        added_count += 1
                        added_records.append(it)
                except Exception as e:
                    print("Lỗi chèn vào qa_data:", e)
        else:
            added_count = len(to_add)
            added_records = to_add

        # 2. Đồng bộ vào data/sample_qa.json
        if persist_to_json and added_records:
            if not json_path:
                json_path = os.path.abspath("data/sample_qa.json")
            try:
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                else:
                    file_data = []

                file_existing = {it["question"].strip().lower() for it in file_data}
                file_added = 0
                for rec in added_records:
                    if rec["question"].strip().lower() not in file_existing:
                        file_data.append({
                            "question": rec["question"],
                            "answer": rec["answer"],
                            "intent": rec["intent"],
                            "tour_id": rec.get("tour_id")
                        })
                        file_existing.add(rec["question"].strip().lower())
                        file_added += 1

                if file_added > 0:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(file_data, f, ensure_ascii=False, indent=2)
                    print(f"-> Đã đồng bộ {file_added} mẫu QA mới vào {json_path}")
            except Exception as e:
                print("Lỗi đồng bộ vào sample_qa.json:", e)

        return {
            "success": True,
            "added_count": added_count,
            "duplicates_skipped": duplicates_skipped,
            "added_records": added_records,
            "message": f"Đã lưu thành công {added_count} mẫu Q&A mới (bỏ qua {duplicates_skipped} mẫu trùng lặp)."
        }

    def retrain_model(self, save_artifacts=True, evaluate=True, verbose=True):
        """
        Kích hoạt tái huấn luyện mạng nơ-ron sâu PyTorch (PyTorchDeepIntentNet)
        và nạp nóng mô hình mới vào Chatbot.
        """
        from ai.train_model import train_model

        if verbose:
            print("\n" + "="*60)
            print("🚀 BẮT ĐẦU TÁI HUẤN LUYỆN MẠNG NƠ-RON SÂU PYTORCH DEEP INTENT NET")
            print("="*60)

        start_time = time.time()
        vectorizer, model = train_model(
            force_retrain=True,
            save_artifacts=save_artifacts,
            evaluate=evaluate,
            verbose=verbose
        )

        duration = round(time.time() - start_time, 2)

        # Nạp nóng lại chatbot
        if verbose:
            print("\n[*] Nạp nóng lại mô hình (Hot-reloading) trong Chatbot...")
        self.bot.reload(force_retrain=False)

        # Lấy thông số đánh giá từ metrics json
        metrics_path = os.path.abspath("ai/saved_models/training_metrics.json")
        eval_metrics = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    eval_metrics = json.load(f)
            except Exception:
                pass

        return {
            "success": model is not None,
            "duration_seconds": duration,
            "metrics": eval_metrics
        }

    def run_full_pipeline(self, num_samples=30, categories=None, dry_run=False, auto_retrain=True, export_path=None, verbose=True):
        """
        THỰC THI TOÀN BỘ QUY TRÌNH TỰ ĐỘNG KHÉP KÍN:
        1. Sinh câu hỏi du lịch.
        2. Chatbot trả lời & thu hoạch dữ liệu.
        3. Kiểm duyệt chất lượng.
        4. Nạp vào MySQL và sample_qa.json (nếu không dry_run).
        5. Tái huấn luyện PyTorch Deep Intent Net & cập nhật 64-D Latent Semantic Embeddings.
        6. Báo cáo kết quả.
        """
        generator = TravelQuestionGenerator()
        existing_set = self.get_existing_questions_set()

        if verbose:
            print(f"[*] Bắt đầu sinh {num_samples} câu hỏi du lịch tự động...")

        questions_data = generator.generate_batch(
            count=num_samples,
            categories=categories,
            ensure_unique=True,
            exclude_questions=existing_set
        )

        if verbose:
            print(f"[*] Đã sinh thành công {len(questions_data)} câu hỏi độc nhất.")
            print(f"[*] Bắt đầu cho Chatbot trả lời và thu hoạch dữ liệu...")

        harvested = self.harvest_answers(questions_data, verbose=verbose)

        valid_count = sum(1 for h in harvested if h["is_valid"])
        invalid_count = len(harvested) - valid_count

        # Xuất file nếu yêu cầu
        if export_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(harvested, f, ensure_ascii=False, indent=2)
                if verbose:
                    print(f"[*] Đã xuất tập dữ liệu thu hoạch vào: {export_path}")
            except Exception as e:
                print("Lỗi xuất file tập dữ liệu:", e)

        ingest_result = {"added_count": 0, "duplicates_skipped": 0}
        retrain_result = {"success": True, "metrics": {}}

        if not dry_run and valid_count > 0:
            if verbose:
                print(f"[*] Nạp {valid_count} mẫu Q&A vào cơ sở tri thức (MySQL & sample_qa.json)...")
            ingest_result = self.ingest_and_sync(harvested, persist_to_db=True, persist_to_json=True)

            if auto_retrain and ingest_result["added_count"] > 0:
                if verbose:
                    print(f"[*] Kích hoạt tái huấn luyện mô hình học sâu...")
                retrain_result = self.retrain_model(save_artifacts=True, evaluate=True, verbose=verbose)
            elif not auto_retrain:
                if verbose:
                    print("[*] Đã bỏ qua bước tái huấn luyện mô hình (auto_retrain=False).")
            else:
                if verbose:
                    print("[*] Không có mẫu mới cần huấn luyện lại.")
        elif dry_run:
            if verbose:
                print("[!] Chế độ DRY-RUN: Bỏ qua bước lưu cơ sở dữ liệu và huấn luyện lại.")

        return {
            "total_generated": len(questions_data),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "harvested_data": harvested,
            "ingest_result": ingest_result,
            "retrain_result": retrain_result
        }
