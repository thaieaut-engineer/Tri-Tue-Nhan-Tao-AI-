from database.db import get_connection


def create_review(user_id, tour_id, rating, comment):
    """
    Thêm đánh giá & nhận xét mới của người dùng cho tour.
    """
    if not (1 <= rating <= 5):
        return False, "Điểm đánh giá phải từ 1 đến 5 sao."

    if not comment or not comment.strip():
        return False, "Nội dung nhận xét không được để trống."

    conn = get_connection()
    if not conn:
        return False, "Không thể kết nối cơ sở dữ liệu."

    try:
        cursor = conn.cursor()
        sql = """
            INSERT INTO reviews (user_id, tour_id, rating, comment)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(sql, (user_id, tour_id, rating, comment.strip()))
        conn.commit()
        return True, cursor.lastrowid
    except Exception as e:
        print("Lỗi tạo đánh giá tour:", e)
        return False, str(e)
    finally:
        cursor.close()
        conn.close()


def get_reviews_by_tour(tour_id):
    """
    Lấy danh sách các nhận xét của một tour kèm thông tin người đánh giá.
    """
    conn = get_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        sql = """
            SELECT r.*, u.full_name, u.username
            FROM reviews r
            JOIN users u ON r.user_id = u.id
            WHERE r.tour_id = %s
            ORDER BY r.id DESC
        """
        cursor.execute(sql, (tour_id,))
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi lấy đánh giá theo tour:", e)
        return []
    finally:
        cursor.close()
        conn.close()


def get_tour_rating_stats(tour_id):
    """
    Tính điểm trung bình và tổng số lượng đánh giá của một tour.
    """
    conn = get_connection()
    if not conn:
        return {"avg_rating": 0.0, "total_reviews": 0}

    try:
        cursor = conn.cursor(dictionary=True)
        sql = """
            SELECT AVG(rating) as avg_rating, COUNT(*) as total_reviews
            FROM reviews
            WHERE tour_id = %s
        """
        cursor.execute(sql, (tour_id,))
        res = cursor.fetchone()
        avg = round(float(res["avg_rating"]), 1) if res and res["avg_rating"] else 5.0
        total = res["total_reviews"] if res else 0
        return {"avg_rating": avg, "total_reviews": total}
    except Exception as e:
        print("Lỗi tính thống kê đánh giá tour:", e)
        return {"avg_rating": 5.0, "total_reviews": 0}
    finally:
        cursor.close()
        conn.close()


def delete_review(review_id):
    """
    Xóa đánh giá (chỉ dành cho Admin).
    """
    conn = get_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM reviews WHERE id = %s", (review_id,))
        conn.commit()
        return True
    except Exception as e:
        print("Lỗi xóa đánh giá:", e)
        return False
    finally:
        cursor.close()
        conn.close()


def get_all_reviews(limit=100):
    """
    Lấy danh sách tất cả các đánh giá trong hệ thống.
    """
    conn = get_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        sql = """
            SELECT r.*, t.name as tour_name, u.full_name, u.username
            FROM reviews r
            JOIN tours t ON r.tour_id = t.id
            JOIN users u ON r.user_id = u.id
            ORDER BY r.id DESC
            LIMIT %s
        """
        cursor.execute(sql, (limit,))
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi lấy danh sách đánh giá toàn hệ thống:", e)
        return []
    finally:
        cursor.close()
        conn.close()
