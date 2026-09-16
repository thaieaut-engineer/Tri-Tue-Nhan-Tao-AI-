from database.db import get_connection


def get_all_qa():
    """
    Lấy toàn bộ dữ liệu hỏi đáp AI kèm tên tour tương ứng.
    """
    connection = get_connection()
    if not connection:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        sql = """
            SELECT 
                q.id,
                q.question,
                q.answer,
                q.intent,
                q.tour_id,
                q.created_at,
                t.name AS tour_name
            FROM qa_data q
            LEFT JOIN tours t ON q.tour_id = t.id
            ORDER BY q.id DESC
        """
        cursor.execute(sql)
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi lấy danh sách QA:", e)
        return []
    finally:
        cursor.close()
        connection.close()


def get_qa_by_id(qa_id):
    """
    Lấy thông tin một câu hỏi đáp theo ID.
    """
    connection = get_connection()
    if not connection:
        return None

    try:
        cursor = connection.cursor(dictionary=True)
        sql = """
            SELECT 
                q.id,
                q.question,
                q.answer,
                q.intent,
                q.tour_id,
                q.created_at,
                t.name AS tour_name
            FROM qa_data q
            LEFT JOIN tours t ON q.tour_id = t.id
            WHERE q.id = %s
        """
        cursor.execute(sql, (qa_id,))
        return cursor.fetchone()
    except Exception as e:
        print("Lỗi lấy QA theo id:", e)
        return None
    finally:
        cursor.close()
        connection.close()


def create_qa(question, answer, intent, tour_id=None):
    """
    Thêm dữ liệu hỏi đáp mới.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = """
            INSERT INTO qa_data (question, answer, intent, tour_id)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(sql, (
            question.strip(),
            answer.strip(),
            intent.strip(),
            tour_id if tour_id else None
        ))
        connection.commit()
        return cursor.lastrowid
    except Exception as e:
        print("Lỗi tạo QA:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def update_qa(qa_id, question, answer, intent, tour_id=None):
    """
    Cập nhật dữ liệu hỏi đáp theo ID.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = """
            UPDATE qa_data
            SET question = %s, answer = %s, intent = %s, tour_id = %s
            WHERE id = %s
        """
        cursor.execute(sql, (
            question.strip(),
            answer.strip(),
            intent.strip(),
            tour_id if tour_id else None,
            qa_id
        ))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi cập nhật QA:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def delete_qa(qa_id):
    """
    Xóa dữ liệu hỏi đáp theo ID.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = "DELETE FROM qa_data WHERE id = %s"
        cursor.execute(sql, (qa_id,))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi xóa QA:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def count_qa():
    """
    Đếm tổng số câu hỏi đáp AI trong hệ thống.
    """
    connection = get_connection()
    if not connection:
        return 0

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM qa_data")
        result = cursor.fetchone()
        return result[0] if result else 0
    except Exception as e:
        print("Lỗi đếm QA:", e)
        return 0
    finally:
        cursor.close()
        connection.close()


def get_distinct_intents():
    """
    Lấy danh sách các ý định (intent) duy nhất hiện có.
    """
    connection = get_connection()
    if not connection:
        return []

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT intent FROM qa_data WHERE intent IS NOT NULL ORDER BY intent ASC")
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        print("Lỗi lấy danh sách intent:", e)
        return []
    finally:
        cursor.close()
        connection.close()
