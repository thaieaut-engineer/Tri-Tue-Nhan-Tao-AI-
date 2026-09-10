import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Đọc các biến môi trường trong file .env
load_dotenv()


def get_connection():
    """
    Tạo kết nối đến cơ sở dữ liệu MySQL.
    """

    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "chatbot_tour"),
            charset="utf8mb4"
        )

        if connection.is_connected():
            return connection

    except Error as e:
        print("Lỗi kết nối MySQL:", e)

    return None