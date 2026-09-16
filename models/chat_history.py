from database.db import get_connection


def save_chat_message(session_id, question, answer):
    """
    Lưu một tin nhắn (câu hỏi và câu trả lời) vào lịch sử phiên trò chuyện.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = """
            INSERT INTO chat_history (session_id, question, answer)
            VALUES (%s, %s, %s)
        """
        cursor.execute(sql, (session_id, question.strip(), answer.strip()))
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
                u.username,
                u.full_name
            FROM chat_history h
            JOIN chat_sessions s ON h.session_id = s.id
            JOIN users u ON s.user_id = u.id
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
