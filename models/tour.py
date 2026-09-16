from database.db import get_connection


def get_all_tours():
    """
    Lấy tất cả tour kèm tên danh mục.
    """
    connection = get_connection()
    if not connection:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        sql = """
            SELECT 
                t.id,
                t.name,
                t.category_id,
                t.destination,
                t.duration,
                t.price,
                t.description,
                t.image,
                c.name AS category_name
            FROM tours t
            LEFT JOIN categories c 
                ON t.category_id = c.id
            ORDER BY t.id DESC
        """
        cursor.execute(sql)
        tours = cursor.fetchall()
        return tours
    except Exception as e:
        print("Lỗi lấy danh sách tour:", e)
        return []
    finally:
        cursor.close()
        connection.close()


def get_tour_by_id(tour_id):
    """
    Lấy thông tin một tour theo ID.
    """
    connection = get_connection()
    if not connection:
        return None

    try:
        cursor = connection.cursor(dictionary=True)
        sql = """
            SELECT 
                t.id,
                t.name,
                t.category_id,
                t.destination,
                t.duration,
                t.price,
                t.description,
                t.image,
                c.name AS category_name
            FROM tours t
            LEFT JOIN categories c 
                ON t.category_id = c.id
            WHERE t.id = %s
        """
        cursor.execute(sql, (tour_id,))
        tour = cursor.fetchone()
        return tour
    except Exception as e:
        print("Lỗi lấy tour:", e)
        return None
    finally:
        cursor.close()
        connection.close()


def get_tour_schedule(tour_id):
    """
    Lấy lịch trình của một tour.
    """
    connection = get_connection()
    if not connection:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        sql = """
            SELECT
                id,
                tour_id,
                day_number,
                location,
                activity,
                description
            FROM tour_schedule
            WHERE tour_id = %s
            ORDER BY day_number ASC, id ASC
        """
        cursor.execute(sql, (tour_id,))
        schedule = cursor.fetchall()
        return schedule
    except Exception as e:
        print("Lỗi lấy lịch trình:", e)
        return []
    finally:
        cursor.close()
        connection.close()


def search_tours(keyword=None, category_id=None, min_price=None, max_price=None, sort_by=None):
    """
    Tìm kiếm và lọc tour theo từ khóa, danh mục, khoảng giá và sắp xếp.
    """
    connection = get_connection()
    if not connection:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT 
                t.id,
                t.name,
                t.category_id,
                t.destination,
                t.duration,
                t.price,
                t.description,
                t.image,
                c.name AS category_name
            FROM tours t
            LEFT JOIN categories c ON t.category_id = c.id
            WHERE 1=1
        """
        params = []

        if keyword:
            kw = f"%{keyword.strip()}%"
            query += " AND (t.name LIKE %s OR t.destination LIKE %s OR t.description LIKE %s)"
            params.extend([kw, kw, kw])

        if category_id:
            query += " AND t.category_id = %s"
            params.append(category_id)

        if min_price is not None:
            query += " AND t.price >= %s"
            params.append(min_price)

        if max_price is not None:
            query += " AND t.price <= %s"
            params.append(max_price)

        if sort_by == "price_asc":
            query += " ORDER BY t.price ASC"
        elif sort_by == "price_desc":
            query += " ORDER BY t.price DESC"
        elif sort_by == "name_asc":
            query += " ORDER BY t.name ASC"
        else:
            query += " ORDER BY t.id DESC"

        cursor.execute(query, tuple(params))
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi tìm kiếm tour:", e)
        return []
    finally:
        cursor.close()
        connection.close()


def create_tour(name, category_id, destination, duration, price, description="", image=""):
    """
    Thêm một tour mới.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = """
            INSERT INTO tours (name, category_id, destination, duration, price, description, image)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            name.strip(),
            category_id if category_id else None,
            destination.strip(),
            duration.strip(),
            price,
            description.strip() if description else "",
            image.strip() if image else None
        ))
        connection.commit()
        return cursor.lastrowid
    except Exception as e:
        print("Lỗi tạo tour:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def update_tour(tour_id, name, category_id, destination, duration, price, description="", image=""):
    """
    Cập nhật tour theo ID.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = """
            UPDATE tours
            SET name = %s, category_id = %s, destination = %s, duration = %s,
                price = %s, description = %s, image = %s
            WHERE id = %s
        """
        cursor.execute(sql, (
            name.strip(),
            category_id if category_id else None,
            destination.strip(),
            duration.strip(),
            price,
            description.strip() if description else "",
            image.strip() if image else None,
            tour_id
        ))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi cập nhật tour:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def delete_tour(tour_id):
    """
    Xóa tour theo ID.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = "DELETE FROM tours WHERE id = %s"
        cursor.execute(sql, (tour_id,))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi xóa tour:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def count_tours():
    """
    Đếm tổng số tour.
    """
    connection = get_connection()
    if not connection:
        return 0

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM tours")
        result = cursor.fetchone()
        return result[0] if result else 0
    except Exception as e:
        print("Lỗi đếm tour:", e)
        return 0
    finally:
        cursor.close()
        connection.close()