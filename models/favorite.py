from database.db import get_connection


def toggle_favorite(user_id, tour_id):
    """
    Thêm hoặc xóa tour khỏi danh sách yêu thích của người dùng.
    Trả về (is_favorited: bool, message: str)
    """
    conn = get_connection()
    if not conn:
        return False, "Không thể kết nối cơ sở dữ liệu."

    try:
        cursor = conn.cursor(dictionary=True)
        # Kiểm tra xem đã yêu thích chưa
        cursor.execute("SELECT id FROM favorites WHERE user_id = %s AND tour_id = %s", (user_id, tour_id))
        row = cursor.fetchone()

        if row:
            # Đã có -> Xóa khỏi danh sách yêu thích
            cursor.execute("DELETE FROM favorites WHERE id = %s", (row["id"],))
            conn.commit()
            return False, "Đã xóa tour khỏi danh sách yêu thích."
        else:
            # Chưa có -> Thêm vào danh sách yêu thích
            cursor.execute("INSERT INTO favorites (user_id, tour_id) VALUES (%s, %s)", (user_id, tour_id))
            conn.commit()
            return True, "Đã lưu tour vào danh sách yêu thích."
    except Exception as e:
        print("Lỗi toggle favorite:", e)
        return False, f"Đã xảy ra lỗi: {e}"
    finally:
        cursor.close()
        conn.close()


def is_tour_favorite(user_id, tour_id):
    """
    Kiểm tra xem tour có được người dùng yêu thích không.
    """
    if not user_id:
        return False

    conn = get_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM favorites WHERE user_id = %s AND tour_id = %s", (user_id, tour_id))
        return cursor.fetchone() is not None
    except Exception as e:
        print("Lỗi kiểm tra favorite:", e)
        return False
    finally:
        cursor.close()
        conn.close()


def get_user_favorite_tour_ids(user_id):
    """
    Lấy danh sách các tour_id mà user đã yêu thích (trả về set để check nhanh O(1)).
    """
    if not user_id:
        return set()

    conn = get_connection()
    if not conn:
        return set()

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT tour_id FROM favorites WHERE user_id = %s", (user_id,))
        rows = cursor.fetchall()
        return {r[0] for r in rows}
    except Exception as e:
        print("Lỗi lấy danh sách favorite tour ids:", e)
        return set()
    finally:
        cursor.close()
        conn.close()


def get_user_favorites(user_id):
    """
    Lấy danh sách chi tiết các tour được người dùng yêu thích.
    """
    conn = get_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        sql = """
            SELECT t.*, c.name as category_name, f.created_at as favorited_at
            FROM favorites f
            JOIN tours t ON f.tour_id = t.id
            LEFT JOIN categories c ON t.category_id = c.id
            WHERE f.user_id = %s
            ORDER BY f.id DESC
        """
        cursor.execute(sql, (user_id,))
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi lấy danh sách tour yêu thích của user:", e)
        return []
    finally:
        cursor.close()
        conn.close()
