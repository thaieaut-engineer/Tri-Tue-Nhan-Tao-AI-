from werkzeug.security import generate_password_hash, check_password_hash
from database.db import get_connection


def get_user_by_id(user_id):
    """
    Lấy thông tin người dùng theo ID.
    """
    connection = get_connection()
    if not connection:
        return None

    try:
        cursor = connection.cursor(dictionary=True)
        sql = "SELECT id, username, full_name, role, created_at FROM users WHERE id = %s"
        cursor.execute(sql, (user_id,))
        return cursor.fetchone()
    except Exception as e:
        print("Lỗi lấy người dùng theo id:", e)
        return None
    finally:
        cursor.close()
        connection.close()


def get_user_by_username(username):
    """
    Lấy thông tin người dùng theo tên đăng nhập (kèm mật khẩu đã hash để xác thực).
    """
    connection = get_connection()
    if not connection:
        return None

    try:
        cursor = connection.cursor(dictionary=True)
        sql = "SELECT id, username, password, full_name, role, created_at FROM users WHERE username = %s"
        cursor.execute(sql, (username.strip(),))
        return cursor.fetchone()
    except Exception as e:
        print("Lỗi lấy người dùng theo username:", e)
        return None
    finally:
        cursor.close()
        connection.close()


def create_user(username, password, full_name, role="user"):
    """
    Tạo tài khoản người dùng mới với mật khẩu được băm an toàn.
    """
    connection = get_connection()
    if not connection:
        return False, "Không thể kết nối cơ sở dữ liệu."

    try:
        cursor = connection.cursor(dictionary=True)
        # Kiểm tra username đã tồn tại chưa
        cursor.execute("SELECT id FROM users WHERE username = %s", (username.strip(),))
        if cursor.fetchone():
            return False, "Tên đăng nhập đã tồn tại."

        hashed_password = generate_password_hash(password)
        sql = """
            INSERT INTO users (username, password, full_name, role)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(sql, (username.strip(), hashed_password, full_name.strip(), role))
        connection.commit()
        return True, cursor.lastrowid
    except Exception as e:
        print("Lỗi tạo người dùng:", e)
        return False, f"Lỗi: {e}"
    finally:
        cursor.close()
        connection.close()


def verify_user(username, password):
    """
    Xác thực thông tin đăng nhập của người dùng.
    Hỗ trợ mật khẩu đã băm (hash) hoặc mật khẩu gốc (nếu có từ seed ban đầu).
    """
    user = get_user_by_username(username)
    if not user:
        return None

    stored_password = user["password"]
    is_valid = False

    try:
        is_valid = check_password_hash(stored_password, password)
    except Exception:
        # Trong trường hợp dữ liệu ban đầu là mật khẩu thường
        if stored_password == password:
            is_valid = True
            # Cập nhật lại mật khẩu thành dạng hash an toàn
            update_user_password(user["id"], password)

    if is_valid:
        return {
            "id": user["id"],
            "username": user["username"],
            "full_name": user["full_name"],
            "role": user["role"]
        }

    return None


def update_user_password(user_id, new_password):
    """
    Cập nhật mật khẩu người dùng.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        hashed = generate_password_hash(new_password)
        cursor.execute("UPDATE users SET password = %s WHERE id = %s", (hashed, user_id))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi đổi mật khẩu:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def get_all_users():
    """
    Lấy danh sách tất cả người dùng (không trả về mật khẩu).
    """
    connection = get_connection()
    if not connection:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        sql = "SELECT id, username, full_name, role, created_at FROM users ORDER BY id DESC"
        cursor.execute(sql)
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi lấy danh sách người dùng:", e)
        return []
    finally:
        cursor.close()
        connection.close()


def update_user_role(user_id, role):
    """
    Cập nhật vai trò người dùng (user hoặc admin).
    """
    if role not in ("user", "admin"):
        return False

    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = "UPDATE users SET role = %s WHERE id = %s"
        cursor.execute(sql, (role, user_id))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi cập nhật vai trò:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def delete_user(user_id):
    """
    Xóa tài khoản người dùng theo ID.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = "DELETE FROM users WHERE id = %s"
        cursor.execute(sql, (user_id,))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi xóa người dùng:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def count_users():
    """
    Đếm tổng số tài khoản người dùng.
    """
    connection = get_connection()
    if not connection:
        return 0

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        result = cursor.fetchone()
        return result[0] if result else 0
    except Exception as e:
        print("Lỗi đếm người dùng:", e)
        return 0
    finally:
        cursor.close()
        connection.close()
