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