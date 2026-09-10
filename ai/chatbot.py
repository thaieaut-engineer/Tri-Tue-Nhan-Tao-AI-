from sklearn.metrics.pairwise import cosine_similarity

from database.db import get_connection
from ai.preprocess import preprocess_text
from ai.train_model import train_model


class Chatbot:

    def __init__(self):

        self.vectorizer = None
        self.model = None

        self.questions = []
        self.answers = []
        self.intents = []
        self.tour_ids = []

        self.load_data()

        self.vectorizer, self.model = train_model()

    def load_data(self):
        """
        Đọc toàn bộ dữ liệu Q&A từ MySQL.
        """

        connection = get_connection()

        if not connection:
            return

        try:

            cursor = connection.cursor(dictionary=True)

            sql = """
                SELECT
                    id,
                    question,
                    answer,
                    intent,
                    tour_id
                FROM qa_data
            """

            cursor.execute(sql)

            data = cursor.fetchall()

            for row in data:

                self.questions.append(
                    preprocess_text(row["question"])
                )

                self.answers.append(
                    row["answer"]
                )

                self.intents.append(
                    row["intent"]
                )

                self.tour_ids.append(
                    row["tour_id"]
                )

        except Exception as e:

            print("Lỗi load dữ liệu chatbot:", e)

        finally:

            cursor.close()
            connection.close()

    def find_best_question(self, question):
        """
        Tìm câu hỏi gần nhất bằng TF-IDF + Cosine Similarity.
        """

        if not self.questions:
            return None

        vectorizer = self.vectorizer

        question_vector = vectorizer.transform(
            [preprocess_text(question)]
        )

        data_vector = vectorizer.transform(
            self.questions
        )

        similarities = cosine_similarity(
            question_vector,
            data_vector
        )[0]

        best_index = similarities.argmax()

        best_score = similarities[best_index]

        return best_index, best_score

    def get_tour(self, tour_id):

        if not tour_id:
            return None

        connection = get_connection()

        if not connection:
            return None

        try:

            cursor = connection.cursor(dictionary=True)

            sql = """
                SELECT
                    id,
                    name,
                    destination,
                    duration,
                    price,
                    description
                FROM tours
                WHERE id = %s
            """

            cursor.execute(
                sql,
                (tour_id,)
            )

            return cursor.fetchone()

        except Exception as e:

            print("Lỗi lấy tour:", e)

            return None

        finally:

            cursor.close()
            connection.close()

    def generate_response(self, question):

        if not question.strip():

            return "Bạn hãy nhập câu hỏi để tôi có thể tư vấn nhé."

        # Nếu model chưa được huấn luyện
        if self.vectorizer is None or self.model is None:

            return "Xin lỗi, hệ thống AI chưa sẵn sàng."

        # -------------------------
        # 1. Phân loại intent
        # -------------------------

        processed_question = preprocess_text(question)

        question_vector = self.vectorizer.transform(
            [processed_question]
        )

        predicted_intent = self.model.predict(
            question_vector
        )[0]

        # -------------------------
        # 2. Tìm câu hỏi gần nhất
        # -------------------------

        result = self.find_best_question(question)

        if result is None:

            return "Xin lỗi, tôi chưa có thông tin phù hợp."

        best_index, score = result

        # Ngưỡng tương đồng
        if score < 0.15:

            return (
                "Xin lỗi, tôi chưa hiểu rõ câu hỏi. "
                "Bạn có thể hỏi về tour, giá tour, "
                "thời gian hoặc địa điểm tham quan."
            )

        answer = self.answers[best_index]

        tour_id = self.tour_ids[best_index]

        # -------------------------
        # 3. Nếu câu hỏi liên quan tour
        # -------------------------

        if tour_id:

            tour = self.get_tour(tour_id)

            if tour:

                if predicted_intent == "tour_price":

                    return (
                        f"💰 {tour['name']} có giá "
                        f"{tour['price']:,.0f} VNĐ."
                    )

                elif predicted_intent == "tour_duration":

                    return (
                        f"⏱ {tour['name']} có thời gian "
                        f"{tour['duration']}."
                    )

                elif predicted_intent == "tour_search":

                    return (
                        f"🌏 Có nhé! "
                        f"{tour['name']} tại {tour['destination']}. "
                        f"Thời gian {tour['duration']}, "
                        f"giá {tour['price']:,.0f} VNĐ."
                    )

                elif predicted_intent == "tour_info":

                    return (
                        f"🌏 {tour['name']}\n\n"
                        f"📍 Điểm đến: {tour['destination']}\n"
                        f"⏱ Thời gian: {tour['duration']}\n"
                        f"💰 Giá: {tour['price']:,.0f} VNĐ\n\n"
                        f"📝 {tour['description']}"
                    )

        return answer