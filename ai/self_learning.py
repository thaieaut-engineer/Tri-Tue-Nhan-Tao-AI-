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
from sklearn.metrics.pairwise import cosine_similarity
from database.db import get_connection
from ai.preprocess import preprocess_text, remove_accents
from models.chat_history import get_unlearned_chat_history, mark_as_learned, count_learned_messages, count_chat_messages
from models.qa_data import create_qa, get_all_qa, count_qa


class ChatSelfLearningEngine:
    def __init__(self, chatbot_instance=None):
        self.chatbot = chatbot_instance

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
        Quét và phân loại các câu hỏi trong lịch sử trò chuyện chưa được học.
        Trả về:
          - auto_learn: Các câu có độ tin cậy cao (>= min_confidence), đủ điều kiện tự học.
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
                "created_at": item["created_at"].strftime('%d/%m/%Y %H:%M') if item["created_at"] else ""
            }

            if confidence >= min_confidence and cosine_score >= 0.30:
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
        1. Chọn các câu hỏi thỏa điều kiện.
        2. Bổ sung vào bảng qa_data với intent và câu trả lời tương ứng.
        3. Đánh dấu is_learned = TRUE trong chat_history.
        4. Tái huấn luyện mô hình PyTorch Deep Learning (chatbot.reload()).
        """
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

        for t in targets:
            q = t["question"]
            ans = t["suggested_answer"] or f"Thông tin tư vấn cho chủ đề {t['predicted_intent']}."
            intent = t["predicted_intent"]
            tour_id = t["tour_id"]

            new_qa_id = create_qa(q, ans, intent, tour_id)
            if new_qa_id:
                learned_history_ids.append(t["id"])
                augmented_samples += 1

        # Đánh dấu trong chat_history
        if learned_history_ids:
            mark_as_learned(learned_history_ids)

        # Tái huấn luyện mạng nơ-ron sâu PyTorch với dữ liệu mới
        bot = self._get_chatbot()
        bot.reload()

        return {
            "success": True,
            "message": f"🎉 AI đã tự học thành công {augmented_samples} mẫu câu hỏi mới từ người dùng!",
            "learned_count": augmented_samples,
            "total_qa_now": count_qa()
        }

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
