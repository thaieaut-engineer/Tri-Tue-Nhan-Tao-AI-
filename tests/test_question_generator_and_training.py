"""
tests/test_question_generator_and_training.py
BỘ KIỂM THỬ TỰ ĐỘNG CHO TRÌNH SINH CÂU HỎI & ĐƯỜNG ỐNG TÁI HUẤN LUYỆN
(UNIT & INTEGRATION TESTS FOR QA GENERATOR & RETRAINING PIPELINE)
"""

import os
import sys
import unittest
import numpy as np

# Đảm bảo import được module từ thư mục gốc
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai.qa_generator import TravelQuestionGenerator, DataHarvestingTrainingPipeline, TEMPLATES
from ai.chatbot import Chatbot
from ai.preprocess import preprocess_text


class TestTravelQuestionGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = TravelQuestionGenerator(seed=42)
        cls.bot = Chatbot()
        cls.pipeline = DataHarvestingTrainingPipeline(chatbot_instance=cls.bot)

    def test_01_supported_categories(self):
        """Kiểm tra danh mục câu hỏi được hỗ trợ đầy đủ 7 phân hệ."""
        cats = self.generator.get_supported_categories()
        expected = ["hotel", "group", "budget", "schedule", "weather", "policy", "overseas"]
        for exp in expected:
            self.assertIn(exp, cats, f"Thiếu danh mục {exp}")

    def test_02_generate_single_across_all_categories(self):
        """Kiểm tra sinh câu hỏi đơn lẻ cho từng chuyên mục."""
        for cat in self.generator.categories:
            item = self.generator.generate_single(category=cat)
            self.assertIsInstance(item, dict)
            self.assertTrue(len(item["question"]) > 10, f"Câu hỏi quá ngắn: {item['question']}")
            self.assertEqual(item["category"], cat)
            self.assertIn("expected_intent", item)
            self.assertIn("metadata", item)

    def test_03_generate_batch_uniqueness(self):
        """Kiểm tra sinh mẻ câu hỏi đảm bảo không bị trùng lặp."""
        batch = self.generator.generate_batch(count=20, ensure_unique=True)
        self.assertEqual(len(batch), 20)

        # Kiểm tra tính độc nhất
        normalized_q = [preprocess_text(it["question"]) for it in batch]
        self.assertEqual(len(normalized_q), len(set(normalized_q)), "Phát hiện câu hỏi trùng lặp trong batch!")

    def test_04_chatbot_answer_harvesting(self):
        """Kiểm tra Chatbot trả lời câu hỏi sinh tự động và thu hoạch phản hồi."""
        # Lấy 1 câu từ 4 nhóm trọng tâm: hotel, group, budget, weather
        sample_questions = [
            self.generator.generate_single(category="hotel"),
            self.generator.generate_single(category="group"),
            self.generator.generate_single(category="budget"),
            self.generator.generate_single(category="weather"),
        ]

        harvested = self.pipeline.harvest_answers(sample_questions, verbose=False)
        self.assertEqual(len(harvested), 4)

        for h in harvested:
            self.assertTrue(h["is_valid"], f"Câu trả lời không đạt: {h['invalid_reasons']}")
            self.assertTrue(len(h["answer"]) >= 30, "Câu trả lời quá ngắn")
            self.assertIsNotNone(h["intent"], "Không nhận diện được intent")
            self.assertGreater(h["confidence"], 0.0, "Độ tin cậy <= 0")
            self.assertGreater(h["latency_ms"], 0.0, "Thời gian phản hồi <= 0")

    def test_05_deduplication_filtering(self):
        """Kiểm tra bộ lọc trùng lặp khi nạp dữ liệu."""
        existing_set = self.pipeline.get_existing_questions_set()
        self.assertGreater(len(existing_set), 100, "Tập câu hỏi có sẵn phải lớn hơn 100 mẫu")

        # Tạo một bản ghi giả lập câu hỏi đã có sẵn
        dummy_harvested = [
            {
                "question": "Xin chào",
                "answer": "Xin chào bạn!",
                "intent": "chao_hoi",
                "tour_id": None,
                "is_valid": True
            }
        ]
        res = self.pipeline.ingest_and_sync(dummy_harvested, persist_to_db=False, persist_to_json=False)
        self.assertEqual(res["added_count"], 0, "Câu hỏi 'Xin chào' đã có sẵn phải bị loại bỏ")
        self.assertEqual(res["duplicates_skipped"], 1)

    def test_06_model_retraining_and_embeddings(self):
        """Kiểm tra đường ống huấn luyện mô hình PyTorch Deep Learning và trích xuất vector 64-D."""
        # Kiểm tra file trọng số và vector nhúng hiện tại
        weights_path = os.path.abspath("ai/saved_models/deep_intent_model.pth")
        embeddings_path = os.path.abspath("ai/saved_models/question_embeddings.npy")
        metrics_path = os.path.abspath("ai/saved_models/training_metrics.json")

        self.assertTrue(os.path.exists(weights_path), "Thiếu file trọng số deep_intent_model.pth")
        self.assertTrue(os.path.exists(embeddings_path), "Thiếu file embeddings question_embeddings.npy")

        embeddings = np.load(embeddings_path)
        self.assertEqual(embeddings.ndim, 2)
        self.assertEqual(embeddings.shape[1], 64, "Không gian vector nhúng ngữ nghĩa phải là 64 chiều (64-D)")


if __name__ == "__main__":
    unittest.main()
