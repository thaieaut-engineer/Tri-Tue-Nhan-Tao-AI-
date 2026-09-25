"""
ai/self_learning.py
Module Continual Learning & Active Pseudo-Labeling cho Chatbot AI:
- Khai thác các câu hỏi thực tế của người dùng từ cơ sở dữ liệu chat_history.
- Sử dụng mạng nơ-ron sâu PyTorch (PyTorchDeepIntentNet) và Complement Naive Bayes
  để đo lường độ tin cậy (Softmax Confidence Probability).
- Tự động gán nhãn giả (Pseudo-Labeling) đối với các câu hỏi đạt độ tự tin cao (>= 80%).
- Phát hiện các câu hỏi mới lạ (Out-of-Distribution, < 50%) đưa vào hàng đợi kiểm duyệt.
- Tự động bổ sung các biến thể câu hỏi vào cơ sở tri thức và tái huấn luyện mô hình.
"""

import os
import json
import threading
from sklearn.metrics.pairwise import cosine_similarity
from database.db import get_connection
from ai.preprocess import preprocess_text, remove_accents
from models.chat_history import (
    get_unlearned_chat_history,
    mark_as_learned,
    mark_as_dismissed,
    count_learned_messages,
    count_chat_messages
)
from models.qa_data import create_qa, get_all_qa, count_qa


def sync_qa_to_sample_file(new_items):
    """Đồng bộ các mẫu QA mới học vào file data/sample_qa.json để đảm bảo tính nhất quán lâu dài."""
    if not new_items:
        return
    json_path = os.path.abspath("data/sample_qa.json")
    try:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        existing_q = {item["question"].strip().lower() for item in data}
        has_new = False
        for it in new_items:
            q_norm = it["question"].strip().lower()
            if q_norm not in existing_q:
                data.append({
                    "question": it["question"],
                    "answer": it["answer"],
                    "intent": it.get("intent", "tu_van_dat_tour"),
                    "tour_id": it.get("tour_id")
                })
                existing_q.add(q_norm)
                has_new = True

        if has_new:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"-> Đã đồng bộ {len(new_items)} mẫu QA mới vào data/sample_qa.json")
    except Exception as e:
        print("Lỗi đồng bộ vào sample_qa.json:", e)


class ChatSelfLearningEngine:
    def __init__(self, chatbot_instance=None):
        self.chatbot = chatbot_instance
        self._is_learning = False
        self._lock = threading.Lock()
        self._counter_since_last_learn = 0

    def _get_chatbot(self):
        if self.chatbot:
            return self.chatbot
        from routes.chat_routes import chatbot
        return chatbot

    def is_spam_or_noise(self, text):
        """Kiểm tra xem câu hỏi có phải là spam, quá ngắn hoặc từ rác không."""
        if not text or len(text.strip()) < 5:
            return True

        text_lower = text.strip().lower()
        words = text_lower.split()
        if len(words) < 2:
            return True

        # Các câu giao tiếp đơn thuần không cần học thành câu hỏi tour
        noise_phrases = {
            "alo", "hello", "hi", "chào", "xin chào", "chào bạn",
            "ok", "oke", "dạ", "vâng", "cảm ơn", "tks", "thanks",
            "tạm biệt", "bye", "test", "123", "abc"
        }
        if text_lower in noise_phrases:
            return True

        return False

    def scan_candidates(self, min_confidence=0.80, limit=200):
        """
        Quét và phân loại các câu hỏi trong lịch sử trò chuyện của tất cả user chưa được học.
        Trả về:
          - auto_learn: Các câu có độ tin cậy cao (>= min_confidence hoặc feedback=1), đủ điều kiện tự học.
          - needs_review: Các câu có độ tin cậy vừa phải (50% - 80%).
          - novel_topics: Các câu lạ (< 50%), nghi vấn là chủ đề/câu hỏi mới.
        """
        bot = self._get_chatbot()
        if not bot or not bot.model or not bot.vectorizer:
            return {
                "success": False,
                "message": "Mô hình Deep Learning chưa sẵn sàng.",
                "auto_learn": [],
                "needs_review": [],
                "novel_topics": []
            }

        unlearned = get_unlearned_chat_history(limit=limit)

        # Tập hợp tất cả các câu hỏi đã có sẵn trong qa_data để tránh trùng lặp
        existing_qa = get_all_qa()
        existing_questions_set = {preprocess_text(q["question"]) for q in existing_qa}

        auto_learn = []
        needs_review = []
        novel_topics = []

        seen_in_batch = set()

        for item in unlearned:
            q_raw = item["question"]
            if self.is_spam_or_noise(q_raw):
                continue

            q_proc = preprocess_text(q_raw)
            if q_proc in existing_questions_set or q_proc in seen_in_batch:
                continue
            seen_in_batch.add(q_proc)

            # Dự đoán Intent và Confidence qua mô hình Deep Learning
            try:
                vec = bot.vectorizer.transform([q_proc])
                prob_arr = bot.model.predict_proba(vec)[0]
                pred_intent = bot.model.predict(vec)[0]
                confidence = float(max(prob_arr))
            except Exception as e:
                print("Lỗi dự đoán trong self_learning:", e)
                continue

            # Tính toán Cosine Similarity để tìm câu trả lời tương ứng tốt nhất
            best_idx, cosine_score = bot.compute_hybrid_cosine_similarity(
                vec,
                {pred_intent: 1.0}
            )

            matched_answer = ""
            matched_tour_id = None
            if best_idx >= 0 and best_idx < len(bot.answers):
                matched_answer = bot.answers[best_idx]
                matched_tour_id = bot.tour_ids[best_idx] if best_idx < len(bot.tour_ids) else None

            user_feedback = item.get("feedback", 0)

            candidate_info = {
                "id": item["id"],
                "question": q_raw,
                "processed_question": q_proc,
                "predicted_intent": pred_intent,
                "confidence": round(confidence, 4),
                "confidence_percent": round(confidence * 100, 1),
                "cosine_score": round(cosine_score, 4),
                "suggested_answer": matched_answer,
                "tour_id": matched_tour_id,
                "feedback": user_feedback,
                "created_at": item["created_at"].strftime('%d/%m/%Y %H:%M') if item["created_at"] else ""
            }

            # Nếu người dùng bấm Like (+1) hoặc độ tin cậy >= min_confidence kết hợp cosine_score >= 0.30
            if (user_feedback == 1 and cosine_score >= 0.28) or (confidence >= min_confidence and cosine_score >= 0.30):
                candidate_info["status"] = "auto_learn"
                auto_learn.append(candidate_info)
            elif confidence >= 0.50:
                candidate_info["status"] = "needs_review"
                needs_review.append(candidate_info)
            else:
                candidate_info["status"] = "novel_topic"
                novel_topics.append(candidate_info)

        return {
            "success": True,
            "total_scanned": len(unlearned),
            "auto_learn": auto_learn,
            "needs_review": needs_review,
            "novel_topics": novel_topics
        }

    def execute_learning(self, candidate_ids=None, min_confidence=0.80):
        """
        Thực thi quy trình tự học:
        1. Chọn các câu hỏi thỏa điều kiện từ lịch sử chat.
        2. Bổ sung vào bảng qa_data với intent và câu trả lời tương ứng.
        3. Đồng bộ vào data/sample_qa.json để duy trì vĩnh viễn.
        4. Đánh dấu is_learned = 1 trong chat_history.
        5. Tái huấn luyện mô hình PyTorch Deep Learning (chatbot.reload()).
        """
        with self._lock:
            scan_result = self.scan_candidates(min_confidence=min_confidence, limit=300)
            if not scan_result.get("success"):
                return scan_result

            all_candidates = (
                scan_result["auto_learn"] +
                scan_result["needs_review"] +
                scan_result["novel_topics"]
            )

            # Lọc danh sách câu hỏi cần học
            targets = []
            if candidate_ids:
                cand_id_set = set(candidate_ids)
                targets = [c for c in all_candidates if c["id"] in cand_id_set]
            else:
                # Mặc định tự học tất cả các câu trong nhóm auto_learn
                targets = scan_result["auto_learn"]

            if not targets:
                return {
                    "success": True,
                    "message": "Không có câu hỏi mới nào đủ tiêu chuẩn tin cậy để học vào lúc này.",
                    "learned_count": 0
                }

            learned_history_ids = []
            augmented_samples = 0
            new_qa_list = []

            for t in targets:
                q = t["question"]
                ans = t["suggested_answer"] or f"Thông tin tư vấn cho chủ đề {t['predicted_intent']}."
                intent = t["predicted_intent"]
                tour_id = t["tour_id"]

                new_qa_id = create_qa(q, ans, intent, tour_id)
                if new_qa_id:
                    learned_history_ids.append(t["id"])
                    augmented_samples += 1
                    new_qa_list.append({
                        "question": q,
                        "answer": ans,
                        "intent": intent,
                        "tour_id": tour_id
                    })

            # Đánh dấu trong chat_history
            if learned_history_ids:
                mark_as_learned(learned_history_ids)

            # Đồng bộ vào file data/sample_qa.json
            if new_qa_list:
                sync_qa_to_sample_file(new_qa_list)

            # Tái huấn luyện mạng nơ-ron sâu PyTorch với dữ liệu mới
            bot = self._get_chatbot()
            bot.reload()

            return {
                "success": True,
                "message": f"🎉 AI đã tự học thành công {augmented_samples} mẫu câu hỏi mới từ người dùng!",
                "learned_count": augmented_samples,
                "total_qa_now": count_qa()
            }

    def learn_custom_question(self, history_id, question, answer, intent, tour_id=None):
        """
        Dành cho Admin duyệt và học một câu hỏi cụ thể với intent và câu trả lời tùy chỉnh.
        """
        with self._lock:
            new_qa_id = create_qa(question, answer, intent, tour_id)
            if new_qa_id:
                mark_as_learned([history_id])
                sync_qa_to_sample_file([{
                    "question": question,
                    "answer": answer,
                    "intent": intent,
                    "tour_id": tour_id
                }])
                bot = self._get_chatbot()
                bot.reload()
                return True
            return False

    def dismiss_questions(self, history_ids):
        """Bỏ qua các câu hỏi không phù hợp để không quét lại lần sau."""
        return mark_as_dismissed(history_ids)

    def trigger_background_auto_learning(self):
        """
        Kích hoạt kiểm tra và tự học ngầm (Continuous Online Learning)
        chạy trên background thread khi người dùng trò chuyện.
        """
        self._counter_since_last_learn += 1
        # Cứ mỗi 3 tin nhắn mới thì kiểm tra học ngầm 1 lần
        if self._counter_since_last_learn < 3:
            return

        self._counter_since_last_learn = 0

        if self._is_learning:
            return

        def _worker():
            self._is_learning = True
            try:
                # Quét và tự học các câu có độ tin cậy >= 85%
                res = self.execute_learning(min_confidence=0.85)
                if res.get("learned_count", 0) > 0:
                    print(f"🤖 [Auto-Learning] {res.get('message')}")
            except Exception as e:
                print("Lỗi chạy auto learning ngầm:", e)
            finally:
                self._is_learning = False

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def get_learning_stats(self):
        """Lấy các chỉ số thống kê về khả năng tự học của hệ thống."""
        bot = self._get_chatbot()
        scan = self.scan_candidates(limit=100)

        return {
            "total_chat_messages": count_chat_messages(),
            "learned_messages": count_learned_messages(),
            "ready_to_learn": len(scan.get("auto_learn", [])),
            "needs_review": len(scan.get("needs_review", [])),
            "novel_topics": len(scan.get("novel_topics", [])),
            "total_qa": count_qa(),
            "deep_learning_active": getattr(bot.model, "has_torch", False) if bot and bot.model else False
        }


# Thể hiện singleton toàn cục
self_learning_engine = ChatSelfLearningEngine()
