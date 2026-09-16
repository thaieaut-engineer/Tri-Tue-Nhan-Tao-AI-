from database.db import get_connection


def create_chat_session(user_id, title="Cuộc trò chuyện mới"):
    """
    Tạo một phiên trò chuyện mới cho người dùng.
    """
    connection = get_connection()
    if not connection:
        return None

    try:
        cursor = connection.cursor()
        sql = "INSERT INTO chat_sessions (user_id, title) VALUES (%s, %s)"
        cursor.execute(sql, (user_id, title.strip()))
        connection.commit()
        return cursor.lastrowid
    except Exception as e:
        print("Lỗi tạo phiên trò chuyện:", e)
        return None
    finally:
        cursor.close()
        connection.close()


def get_user_sessions(user_id):
    """
    Lấy danh sách tất cả các phiên trò chuyện của một người dùng.
    """
    connection = get_connection()
    if not connection:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        sql = """
            SELECT 
                s.id,
                s.user_id,
                s.title,
                s.created_at,
                s.updated_at,
                COUNT(h.id) AS message_count
            FROM chat_sessions s
            LEFT JOIN chat_history h ON s.id = h.session_id
            WHERE s.user_id = %s
            GROUP BY s.id, s.user_id, s.title, s.created_at, s.updated_at
            ORDER BY s.updated_at DESC
        """
        cursor.execute(sql, (user_id,))
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi lấy danh sách phiên trò chuyện:", e)
        return []
    finally:
        cursor.close()
        connection.close()


def get_session_by_id(session_id):
    """
    Lấy thông tin một phiên trò chuyện theo ID.
    """
    connection = get_connection()
    if not connection:
        return None

    try:
        cursor = connection.cursor(dictionary=True)
        sql = """
            SELECT s.id, s.user_id, s.title, s.created_at, s.updated_at, u.username, u.full_name
            FROM chat_sessions s
            LEFT JOIN users u ON s.user_id = u.id
            WHERE s.id = %s
        """
        cursor.execute(sql, (session_id,))
        return cursor.fetchone()
    except Exception as e:
        print("Lỗi lấy phiên trò chuyện:", e)
        return None
    finally:
        cursor.close()
        connection.close()


def update_session_title(session_id, title):
    """
    Cập nhật tiêu đề phiên trò chuyện.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = "UPDATE chat_sessions SET title = %s WHERE id = %s"
        cursor.execute(sql, (title.strip(), session_id))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi cập nhật tiêu đề phiên chat:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def delete_session(session_id):
    """
    Xóa phiên trò chuyện theo ID (kèm tự động xóa lịch sử nhờ CASCADE).
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = "DELETE FROM chat_sessions WHERE id = %s"
        cursor.execute(sql, (session_id,))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi xóa phiên chat:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def count_sessions():
    """
    Đếm tổng số phiên trò chuyện trong toàn hệ thống.
    """
    connection = get_connection()
    if not connection:
        return 0

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM chat_sessions")
        result = cursor.fetchone()
        return result[0] if result else 0
    except Exception as e:
        print("Lỗi đếm phiên chat:", e)
        return 0
    finally:
        cursor.close()
        connection.close()
