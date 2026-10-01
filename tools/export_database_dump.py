#!/usr/bin/env python3
"""
export_database_dump.py
Xuất toàn bộ cấu trúc và dữ liệu của cơ sở dữ liệu chatbot_tour ra file SQL hoàn chỉnh (chatbot_tour_full.sql).
Bao gồm:
- Toàn bộ 10 bảng chuẩn hóa
- 6 danh mục categories
- 20 tour du lịch chi tiết
- 52 lịch trình tour theo ngày
- 1.290 câu hỏi đáp tri thức AI & Deep Learning (qa_data)
- Tài khoản người dùng mẫu và Admin
- Phiên hội thoại & lịch sử trò chuyện
"""

import os
import sys
import datetime
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mysecretpassword")
DB_NAME = os.getenv("DB_NAME", "chatbot_tour")

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "chatbot_tour_full.sql")

TABLES_ORDER = [
    "categories",
    "tours",
    "users",
    "tour_schedule",
    "qa_data",
    "chat_sessions",
    "chat_history",
    "bookings",
    "reviews",
    "favorites"
]

def escape_sql_val(val):
    if val is None:
        return "NULL"
    elif isinstance(val, (int, float)):
        return str(val)
    elif isinstance(val, (datetime.datetime, datetime.date, datetime.time)):
        return f"'{val}'"
    elif isinstance(val, (bytes, bytearray)):
        return f"X'{val.hex()}'"
    elif isinstance(val, bool):
        return "1" if val else "0"
    else:
        s = str(val)
        s = s.replace("\\", "\\\\").replace("'", "\\'").replace("\r", "\\r").replace("\n", "\\n").replace("\0", "\\0")
        return f"'{s}'"

def main():
    print(f"[*] Đang kết nối tới MySQL: {DB_HOST}:{DB_PORT}, database={DB_NAME}, user={DB_USER}...")
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset="utf8mb4"
        )
    except Exception as e:
        print(f"Lỗi kết nối với user {DB_USER}: {e}. Đang thử với kenny...")
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user="kenny",
            password="123456",
            database=DB_NAME,
            charset="utf8mb4"
        )

    cur = conn.cursor()
    cur.execute("SELECT VERSION();")
    server_version = cur.fetchone()[0]

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out = []
    out.append("-- ============================================================================")
    out.append("-- TOURAI - HỆ THỐNG CƠ SỞ DỮ LIỆU ĐẦY ĐỦ (FULL SQL DUMP)")
    out.append(f"-- Máy chủ: MySQL {server_version}")
    out.append(f"-- Cơ sở dữ liệu: {DB_NAME}")
    out.append(f"-- Thời gian xuất bản: {now_str}")
    out.append("-- Bao gồm: 10 Bảng chuẩn hóa, 20 Tours, 52 Lịch trình ngày, 1.290 Q&A Tri thức AI")
    out.append("-- ============================================================================\n")

    out.append("SET NAMES utf8mb4;")
    out.append("SET FOREIGN_KEY_CHECKS = 0;")
    out.append("SET SQL_MODE = 'NO_AUTO_VALUE_ON_ZERO';")
    out.append("SET time_zone = '+00:00';\n")

    out.append("-- ----------------------------------------------------------------------------")
    out.append(f"-- 1. KHỞI TẠO CƠ SỞ DỮ LIỆU `{DB_NAME}`")
    out.append("-- ----------------------------------------------------------------------------")
    out.append(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    out.append(f"USE `{DB_NAME}`;\n")

    for tbl in TABLES_ORDER:
        print(f"[*] Đang xử lý bảng: {tbl}...")
        cur.execute(f"SHOW CREATE TABLE `{tbl}`;")
        res = cur.fetchone()
        if not res:
            continue
        create_table_sql = res[1]

        out.append("-- ----------------------------------------------------------------------------")
        out.append(f"-- CẤU TRÚC VÀ DỮ LIỆU BẢNG: `{tbl}`")
        out.append("-- ----------------------------------------------------------------------------")
        out.append(f"DROP TABLE IF EXISTS `{tbl}`;")
        out.append(create_table_sql + ";\n")

        # Fetch columns
        cur.execute(f"SHOW COLUMNS FROM `{tbl}`;")
        cols = [f"`{col[0]}`" for col in cur.fetchall()]
        cols_str = ", ".join(cols)

        # Fetch data
        cur.execute(f"SELECT * FROM `{tbl}`;")
        rows = cur.fetchall()
        print(f"    -> Đã đọc {len(rows)} bản ghi.")

        if rows:
            out.append(f"-- Dữ liệu bảng `{tbl}` ({len(rows)} bản ghi)")
            out.append(f"LOCK TABLES `{tbl}` WRITE;")
            batch_size = 50
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                values_clauses = []
                for r in batch:
                    vals = [escape_sql_val(v) for v in r]
                    values_clauses.append(f"({', '.join(vals)})")
                
                insert_stmt = f"INSERT INTO `{tbl}` ({cols_str}) VALUES\n  " + ",\n  ".join(values_clauses) + ";"
                out.append(insert_stmt)
            out.append("UNLOCK TABLES;\n")
        else:
            out.append(f"-- Bảng `{tbl}` hiện chưa có dữ liệu mẫu ban đầu.\n")

    out.append("SET FOREIGN_KEY_CHECKS = 1;\n")
    out.append("-- ============================================================================")
    out.append("-- KẾT THÚC BẢN DUMP CƠ SỞ DỮ LIỆU TOURAI")
    out.append("-- ============================================================================")

    full_sql = "\n".join(out)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_sql)

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n✔ ĐÃ XUẤT THÀNH CÔNG: {OUTPUT_FILE}")
    print(f"✔ Kích thước file: {size_mb:.2f} MB")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
