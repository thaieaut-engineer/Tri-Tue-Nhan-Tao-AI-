from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from database.db import get_connection
from ai.preprocess import preprocess_text


def load_training_data():
    """
    Lấy dữ liệu câu hỏi và intent từ MySQL.
    """

    connection = get_connection()

    if not connection:
        return [], []

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

        questions = []
        intents = []

        for row in data:
            questions.append(
                preprocess_text(row["question"])
            )

            intents.append(
                row["intent"]
            )

        return questions, intents

    except Exception as e:
        print("Lỗi lấy dữ liệu huấn luyện:", e)
        return [], []

    finally:
        cursor.close()
        connection.close()


def train_model():
    """
    Huấn luyện mô hình Naive Bayes.
    """

    questions, intents = load_training_data()

    if len(questions) < 2:
        print("Không đủ dữ liệu để huấn luyện AI.")
        return None, None

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(questions)

    model = MultinomialNB()

    model.fit(X, intents)

    return vectorizer, model