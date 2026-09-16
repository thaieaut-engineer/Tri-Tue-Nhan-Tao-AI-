from database.db import get_connection


def get_all_categories():
    """
    Lấy tất cả danh mục tour kèm số lượng tour trong mỗi danh mục.
    """
    connection = get_connection()
    if not connection:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        sql = """
            SELECT 
                c.id, 
                c.name, 
                c.description, 
                c.created_at,
                COUNT(t.id) AS tour_count
            FROM categories c
            LEFT JOIN tours t ON c.id = t.category_id
            GROUP BY c.id, c.name, c.description, c.created_at
            ORDER BY c.id ASC
        """
        cursor.execute(sql)
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi lấy danh sách danh mục:", e)
        return []
    finally:
        cursor.close()
        connection.close()


def get_category_by_id(category_id):
    """
    Lấy thông tin một danh mục theo ID.
    """
    connection = get_connection()
    if not connection:
        return None

    try:
        cursor = connection.cursor(dictionary=True)
        sql = "SELECT id, name, description, created_at FROM categories WHERE id = %s"
        cursor.execute(sql, (category_id,))
        return cursor.fetchone()
    except Exception as e:
        print("Lỗi lấy danh mục:", e)
        return None
    finally:
        cursor.close()
        connection.close()


def create_category(name, description=""):
    """
    Thêm một danh mục mới.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = "INSERT INTO categories (name, description) VALUES (%s, %s)"
        cursor.execute(sql, (name.strip(), description.strip() if description else ""))
        connection.commit()
        return cursor.lastrowid
    except Exception as e:
        print("Lỗi tạo danh mục:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def update_category(category_id, name, description=""):
    """
    Cập nhật danh mục theo ID.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = "UPDATE categories SET name = %s, description = %s WHERE id = %s"
        cursor.execute(sql, (name.strip(), description.strip() if description else "", category_id))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi cập nhật danh mục:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def delete_category(category_id):
    """
    Xóa một danh mục theo ID.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = "DELETE FROM categories WHERE id = %s"
        cursor.execute(sql, (category_id,))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi xóa danh mục:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def count_categories():
    """
    Đếm tổng số danh mục.
    """
    connection = get_connection()
    if not connection:
        return 0

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM categories")
        result = cursor.fetchone()
        return result[0] if result else 0
    except Exception as e:
        print("Lỗi đếm danh mục:", e)
        return 0
    finally:
        cursor.close()
        connection.close()
