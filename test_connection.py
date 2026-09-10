from database.db import get_connection


connection = get_connection()

if connection:
    print("Kết nối MySQL thành công!")

    cursor = connection.cursor()
    cursor.execute("SELECT DATABASE();")

    result = cursor.fetchone()
    print("Database hiện tại:", result[0])

    cursor.close()
    connection.close()

    print("Đã đóng kết nối.")
else:
    print("Kết nối MySQL thất bại!")