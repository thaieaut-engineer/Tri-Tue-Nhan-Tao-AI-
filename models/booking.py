from database.db import get_connection


def create_booking(user_id, tour_id, full_name, phone, email, start_date, adults=1, children=0, total_price=0, note=None):
    """
    Tạo đơn đặt tour mới trong cơ sở dữ liệu.
    """
    conn = get_connection()
    if not conn:
        return None

    try:
        cursor = conn.cursor()
        sql = """
            INSERT INTO bookings (user_id, tour_id, full_name, phone, email, start_date, adults, children, total_price, note, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending')
        """
        cursor.execute(sql, (
            user_id, tour_id, full_name.strip(), phone.strip(), email.strip(),
            start_date, adults, children, total_price, (note or "").strip()
        ))
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        print("Lỗi tạo đơn đặt tour:", e)
        return None
    finally:
        cursor.close()
        conn.close()


def get_user_bookings(user_id):
    """
    Lấy danh sách các đơn đặt tour của một người dùng cụ thể.
    """
    conn = get_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        sql = """
            SELECT b.*, t.name as tour_name, t.destination, t.duration, t.image as tour_image, t.price as tour_price
            FROM bookings b
            JOIN tours t ON b.tour_id = t.id
            WHERE b.user_id = %s
            ORDER BY b.id DESC
        """
        cursor.execute(sql, (user_id,))
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi lấy đơn đặt tour của user:", e)
        return []
    finally:
        cursor.close()
        conn.close()


def get_all_bookings(status=None, search=None):
    """
    Lấy toàn bộ danh sách đơn đặt tour (dành cho Admin quản lý).
    Hỗ trợ lọc theo trạng thái và tìm kiếm theo họ tên hoặc số điện thoại.
    """
    conn = get_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        sql = """
            SELECT b.*, t.name as tour_name, t.destination, t.duration, u.username
            FROM bookings b
            JOIN tours t ON b.tour_id = t.id
            LEFT JOIN users u ON b.user_id = u.id
            WHERE 1=1
        """
        params = []

        if status:
            sql += " AND b.status = %s"
            params.append(status)

        if search:
            sql += " AND (b.full_name LIKE %s OR b.phone LIKE %s OR b.email LIKE %s OR t.name LIKE %s)"
            term = f"%{search}%"
            params.extend([term, term, term, term])

        sql += " ORDER BY b.id DESC"
        cursor.execute(sql, tuple(params))
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi lấy toàn bộ đơn đặt tour:", e)
        return []
    finally:
        cursor.close()
        conn.close()


def get_booking_by_id(booking_id):
    """
    Lấy thông tin chi tiết một đơn đặt tour theo ID.
    """
    conn = get_connection()
    if not conn:
        return None

    try:
        cursor = conn.cursor(dictionary=True)
        sql = """
            SELECT b.*, t.name as tour_name, t.destination, t.duration, t.price as tour_price, u.username
            FROM bookings b
            JOIN tours t ON b.tour_id = t.id
            LEFT JOIN users u ON b.user_id = u.id
            WHERE b.id = %s
        """
        cursor.execute(sql, (booking_id,))
        return cursor.fetchone()
    except Exception as e:
        print("Lỗi lấy đơn đặt tour theo id:", e)
        return None
    finally:
        cursor.close()
        conn.close()


def update_booking_status(booking_id, status):
    """
    Cập nhật trạng thái đơn đặt tour (pending, confirmed, completed, cancelled).
    """
    valid_statuses = ('pending', 'confirmed', 'completed', 'cancelled')
    if status not in valid_statuses:
        return False

    conn = get_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()
        sql = "UPDATE bookings SET status = %s WHERE id = %s"
        cursor.execute(sql, (status, booking_id))
        conn.commit()
        return True
    except Exception as e:
        print("Lỗi cập nhật trạng thái đơn tour:", e)
        return False
    finally:
        cursor.close()
        conn.close()


def cancel_booking_by_user(booking_id, user_id):
    """
    Người dùng tự hủy đơn khi đơn vẫn đang ở trạng thái 'pending'.
    """
    conn = get_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()
        sql = "UPDATE bookings SET status = 'cancelled' WHERE id = %s AND user_id = %s AND status = 'pending'"
        cursor.execute(sql, (booking_id, user_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print("Lỗi người dùng hủy đơn tour:", e)
        return False
    finally:
        cursor.close()
        conn.close()


def delete_booking(booking_id):
    """
    Xóa đơn đặt tour khỏi CSDL (chỉ dành cho Admin).
    """
    conn = get_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()
        sql = "DELETE FROM bookings WHERE id = %s"
        cursor.execute(sql, (booking_id,))
        conn.commit()
        return True
    except Exception as e:
        print("Lỗi xóa đơn đặt tour:", e)
        return False
    finally:
        cursor.close()
        conn.close()


def count_bookings(status=None):
    """
    Đếm tổng số đơn đặt tour (hoặc theo trạng thái).
    """
    conn = get_connection()
    if not conn:
        return 0

    try:
        cursor = conn.cursor()
        if status:
            cursor.execute("SELECT COUNT(*) FROM bookings WHERE status = %s", (status,))
        else:
            cursor.execute("SELECT COUNT(*) FROM bookings")
        res = cursor.fetchone()
        return res[0] if res else 0
    except Exception as e:
        print("Lỗi đếm đơn đặt tour:", e)
        return 0
    finally:
        cursor.close()
        conn.close()


def get_total_revenue():
    """
    Tính tổng doanh thu từ các đơn tour đã xác nhận hoặc hoàn thành.
    """
    conn = get_connection()
    if not conn:
        return 0

    try:
        cursor = conn.cursor()
        sql = "SELECT SUM(total_price) FROM bookings WHERE status IN ('confirmed', 'completed')"
        cursor.execute(sql)
        res = cursor.fetchone()
        return float(res[0]) if res and res[0] else 0.0
    except Exception as e:
        print("Lỗi tính tổng doanh thu:", e)
        return 0.0
    finally:
        cursor.close()
        conn.close()
