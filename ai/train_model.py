import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from database.db import get_connection
from ai.preprocess import preprocess_text


def load_training_data():
    """
    Lấy dữ liệu câu hỏi và intent từ MySQL hoặc fallback file sample_qa.json.
    """
    questions = []
    intents = []

    # 1. Thử lấy từ cơ sở dữ liệu MySQL
    connection = get_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            sql = """
                SELECT question, intent
                FROM qa_data
                WHERE question IS NOT NULL
                  AND intent IS NOT NULL
            """
            cursor.execute(sql)
            data = cursor.fetchall()
            for row in data:
                questions.append(preprocess_text(row["question"]))
                intents.append(row["intent"])
        except Exception as e:
            print("Lỗi lấy dữ liệu huấn luyện từ DB:", e)
        finally:
            cursor.close()
            connection.close()

    # 2. Nếu trong DB chưa có đủ dữ liệu, dùng file sample_qa.json
    if len(questions) < 2:
        json_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_qa.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    sample_data = json.load(f)
                    for item in sample_data:
                        questions.append(preprocess_text(item["question"]))
                        intents.append(item["intent"])
            except Exception as e:
                print("Lỗi đọc sample_qa.json:", e)

    return questions, intents


def train_model():
    """
    HUẤN LUYỆN MÔ HÌNH AI PHÂN LOẠI Ý ĐỊNH (INTENT CLASSIFICATION):
    - TfidfVectorizer nâng cao với 1-gram, 2-gram, 3-gram và log-scaling (sublinear_tf=True).
    - Naive Bayes nâng cao (ComplementNB với smoothing alpha=0.5), tối ưu hóa đặc thù cho
      dữ liệu văn bản và giải quyết hiện tượng mất cân bằng lớp (class imbalance).
    """
    questions, intents = load_training_data()

    if len(questions) < 2:
        print("Không đủ dữ liệu để huấn luyện AI.")
        return None, None

    # Vectorizer với n-gram mở rộng lên đến 3 từ ghép liên tiếp
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1
    )

    X = vectorizer.fit_transform(questions)

    # Sử dụng Complement Naive Bayes (biến thể nâng cao của MultinomialNB)
    try:
        model = ComplementNB(alpha=0.5)
        model.fit(X, intents)
    except Exception:
        # Fallback sang MultinomialNB truyền thống nếu môi trường yêu cầu
        model = MultinomialNB(alpha=0.1)
        model.fit(X, intents)

    return vectorizer, model