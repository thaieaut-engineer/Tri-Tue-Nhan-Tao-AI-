from database.db import get_connection


def save_chat_message(session_id, question, answer, intent=None, confidence=None):
    """
    Lưu một tin nhắn (câu hỏi và câu trả lời) vào lịch sử phiên trò chuyện
    kèm intent và điểm tin cậy (confidence) phục vụ Continual Learning.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = """
            INSERT INTO chat_history (session_id, question, answer, intent, confidence, is_learned)
            VALUES (%s, %s, %s, %s, %s, FALSE)
        """
        cursor.execute(sql, (
            session_id,
            question.strip(),
            answer.strip(),
            intent.strip() if intent else None,
            float(confidence) if confidence is not None else None
        ))
        connection.commit()
        return cursor.lastrowid
    except Exception as e:
        print("Lỗi lưu lịch sử chat:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def get_chat_history_by_session(session_id):
    """
    Lấy toàn bộ lịch sử tin nhắn trong một phiên trò chuyện theo thứ tự thời gian.
    """
    connection = get_connection()
    if not connection:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        sql = """
            SELECT id, session_id, question, answer, created_at
            FROM chat_history
            WHERE session_id = %s
            ORDER BY created_at ASC, id ASC
        """
        cursor.execute(sql, (session_id,))
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi lấy lịch sử chat:", e)
        return []
    finally:
        cursor.close()
        connection.close()


def get_all_chat_logs(limit=100):
    """
    Lấy danh sách câu hỏi & trả lời gần nhất trên toàn hệ thống (dành cho Admin).
    """
    connection = get_connection()
    if not connection:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        sql = """
            SELECT 
                h.id,
                h.session_id,
                h.question,
                h.answer,
                h.created_at,
                s.title AS session_title,
                COALESCE(u.username, 'Khách vãng lai') AS username,
                COALESCE(u.full_name, 'Khách vãng lai') AS full_name
            FROM chat_history h
            JOIN chat_sessions s ON h.session_id = s.id
            LEFT JOIN users u ON s.user_id = u.id
            ORDER BY h.created_at DESC
            LIMIT %s
        """
        cursor.execute(sql, (limit,))
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi lấy log chat:", e)
        return []
    finally:
        cursor.close()
        connection.close()


def count_chat_messages():
    """
    Đếm tổng số tin nhắn được hỏi đáp qua chatbot.
    """
    connection = get_connection()
    if not connection:
        return 0

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM chat_history")
        result = cursor.fetchone()
        return result[0] if result else 0
    except Exception as e:
        print("Lỗi đếm tin nhắn chat:", e)
        return 0
    finally:
        cursor.close()
        connection.close()


def get_unlearned_chat_history(limit=500):
    """
    Lấy danh sách các câu hỏi từ người dùng chưa được mô hình AI học (is_learned = FALSE).
    """
    conn = get_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        sql = """
            SELECT id, session_id, question, answer, intent, confidence, feedback, created_at
            FROM chat_history
            WHERE (is_learned = FALSE OR is_learned = 0) AND LENGTH(TRIM(question)) >= 6
            ORDER BY id DESC
            LIMIT %s
        """
        cursor.execute(sql, (limit,))
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi lấy tin nhắn chưa học:", e)
        return []
    finally:
        cursor.close()
        conn.close()


def update_chat_feedback(message_id, feedback_val):
    """
    Cập nhật đánh giá của người dùng cho câu trả lời (+1: hữu ích/hài lòng, -1: không hài lòng).
    """
    conn = get_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()
        sql = "UPDATE chat_history SET feedback = %s WHERE id = %s"
        cursor.execute(sql, (int(feedback_val), int(message_id)))
        conn.commit()
        return True
    except Exception as e:
        print("Lỗi cập nhật feedback:", e)
        return False
    finally:
        cursor.close()
        conn.close()


def mark_as_learned(history_ids):
    """
    Đánh dấu danh sách các câu hỏi đã được tích hợp thành công vào tập huấn luyện của AI.
    """
    if not history_ids:
        return 0

    conn = get_connection()
    if not conn:
        return 0

    try:
        cursor = conn.cursor()
        format_strings = ','.join(['%s'] * len(history_ids))
        sql = f"UPDATE chat_history SET is_learned = 1 WHERE id IN ({format_strings})"
        cursor.execute(sql, tuple(history_ids))
        conn.commit()
        return cursor.rowcount
    except Exception as e:
        print("Lỗi đánh dấu tin nhắn đã học:", e)
        return 0
    finally:
        cursor.close()
        conn.close()


def mark_as_dismissed(history_ids):
    """
    Đánh dấu bỏ qua các câu hỏi không phù hợp để không quét lại lần sau (is_learned = 2).
    """
    if not history_ids:
        return 0

    conn = get_connection()
    if not conn:
        return 0

    try:
        cursor = conn.cursor()
        format_strings = ','.join(['%s'] * len(history_ids))
        sql = f"UPDATE chat_history SET is_learned = 2 WHERE id IN ({format_strings})"
        cursor.execute(sql, tuple(history_ids))
        conn.commit()
        return cursor.rowcount
    except Exception as e:
        print("Lỗi bỏ qua tin nhắn:", e)
        return 0
    finally:
        cursor.close()
        conn.close()


def count_learned_messages():
    """
    Đếm tổng số câu hỏi từ lịch sử chat đã được mô hình AI tự học thành công.
    """
    conn = get_connection()
    if not conn:
        return 0

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chat_history WHERE is_learned = 1")
        res = cursor.fetchone()
        return res[0] if res else 0
    except Exception as e:
        print("Lỗi đếm số tin nhắn đã học:", e)
        return 0
    finally:
        cursor.close()
        conn.close()
