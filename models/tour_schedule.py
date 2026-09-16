from database.db import get_connection


def get_tour_schedule(tour_id):
    """
    Lấy toàn bộ lịch trình chi tiết của một tour theo ID sắp xếp theo ngày.
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
        return cursor.fetchall()
    except Exception as e:
        print("Lỗi lấy lịch trình tour:", e)
        return []
    finally:
        cursor.close()
        connection.close()


def add_schedule_item(tour_id, day_number, location, activity, description=""):
    """
    Thêm một mục lịch trình mới cho tour.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = """
            INSERT INTO tour_schedule (tour_id, day_number, location, activity, description)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (tour_id, day_number, location.strip(), activity.strip(), description.strip() if description else ""))
        connection.commit()
        return cursor.lastrowid
    except Exception as e:
        print("Lỗi thêm lịch trình:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def update_schedule_item(schedule_id, day_number, location, activity, description=""):
    """
    Cập nhật mục lịch trình theo ID.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = """
            UPDATE tour_schedule
            SET day_number = %s, location = %s, activity = %s, description = %s
            WHERE id = %s
        """
        cursor.execute(sql, (day_number, location.strip(), activity.strip(), description.strip() if description else "", schedule_id))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi cập nhật lịch trình:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def delete_schedule_item(schedule_id):
    """
    Xóa một mục lịch trình theo ID.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = "DELETE FROM tour_schedule WHERE id = %s"
        cursor.execute(sql, (schedule_id,))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi xóa lịch trình:", e)
        return False
    finally:
        cursor.close()
        connection.close()


def delete_schedule_by_tour(tour_id):
    """
    Xóa tất cả lịch trình của một tour.
    """
    connection = get_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        sql = "DELETE FROM tour_schedule WHERE tour_id = %s"
        cursor.execute(sql, (tour_id,))
        connection.commit()
        return True
    except Exception as e:
        print("Lỗi xóa lịch trình của tour:", e)
        return False
    finally:
        cursor.close()
        connection.close()
