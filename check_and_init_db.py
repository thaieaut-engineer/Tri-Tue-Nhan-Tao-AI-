#!/usr/bin/env python3
"""
check_and_init_db.py
Script kiểm tra và khởi tạo cơ sở dữ liệu MySQL/MariaDB cho dự án TourAI.
Hỗ trợ đầy đủ:
 - Docker MySQL + DBeaver / CloudBeaver
 - Windows Native Service (MySQL, MySQL80, MariaDB)
 - Windows XAMPP, Laragon, WampServer
 - Linux Native Service (systemd)
"""

import os
import sys
import json
import socket
import platform
import subprocess
from pathlib import Path
from werkzeug.security import generate_password_hash


# =======================================================
# MÀU SẮC TERMINAL
# =======================================================
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


if platform.system() == "Windows":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


def print_step(title):
    print(f"\n{Colors.CYAN}{Colors.BOLD}[*] {title}{Colors.RESET}")


def print_success(msg):
    print(f"  {Colors.GREEN}✔ {msg}{Colors.RESET}")


def print_warning(msg):
    print(f"  {Colors.YELLOW}⚠ {msg}{Colors.RESET}")


def print_error(msg):
    print(f"  {Colors.RED}✖ {msg}{Colors.RESET}")


def print_info(msg):
    print(f"  {Colors.BLUE}ℹ {msg}{Colors.RESET}")


ENV_FILE = Path(__file__).resolve().parent / ".env"
SCHEMA_FILE = Path(__file__).resolve().parent / "database" / "schema.sql"
SAMPLE_QA_FILE = Path(__file__).resolve().parent / "data" / "sample_qa.json"


# =======================================================
# 1. ĐỌC VÀ CẬP NHẬT FILE .ENV
# =======================================================
def load_env_config():
    config = {
        "DB_HOST": "127.0.0.1",
        "DB_PORT": "3306",
        "DB_USER": "root",
        "DB_PASSWORD": "",
        "DB_NAME": "chatbot_tour",
        "SECRET_KEY": "chatbot_tour_secret_key"
    }

    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    config[key.strip()] = val.strip()
    return config


def update_env_file(key, value):
    """Cập nhật hoặc thêm biến vào file .env"""
    if not ENV_FILE.exists():
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")
        return

    lines = []
    found = False
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith(f"{key}="):
                lines.append(f"{key}={value}\n")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}\n")

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)


# =======================================================
# 2. KIỂM TRA CỔNG TCP (PORT LISTENING)
# =======================================================
def is_port_open(host, port, timeout=1.5):
    """Kiểm tra cổng TCP của MySQL Server"""
    try:
        if host in ("localhost", "0.0.0.0"):
            host = "127.0.0.1"
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, int(port)))
        sock.close()
        return result == 0
    except Exception:
        return False


# =======================================================
# 3. QUÉT VÀ NHẬN DIỆN MÔI TRƯỜNG HỆ THỐNG
# =======================================================
def check_system_environment():
    os_name = platform.system()
    found_info = []

    print_info(f"Hệ điều hành hiện tại: {os_name} ({platform.release()})")

    # A. Kiểm tra Docker (cả Windows & Linux)
    try:
        res = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            found_info.append(f"Docker đã cài đặt: {res.stdout.strip()}")

            ps_res = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}} | {{.Image}} | {{.Status}}"],
                                    capture_output=True, text=True, timeout=3)
            if ps_res.returncode == 0 and ps_res.stdout.strip():
                containers = ps_res.stdout.strip().split("\n")
                found_info.append("Danh sách Docker containers liên quan:")
                for c in containers:
                    if any(kw in c.lower() for kw in ["mysql", "maria", "dbeaver", "database", "db"]):
                        found_info.append(f"   -> [Container] {c}")
    except Exception:
        pass

    # B. Tìm kiếm file docker-compose.yml
    home_dir = Path.home()
    compose_candidates = [
        Path("docker-compose.yml"),
        home_dir / "mysql-dbeaver" / "docker-compose.yml",
        home_dir / "docker" / "docker-compose.yml",
        home_dir / "mysql" / "docker-compose.yml"
    ]
    for candidate in compose_candidates:
        if candidate.exists():
            found_info.append(f"Tìm thấy cấu hình Docker Compose: {candidate}")

    # C. Kiểm tra dịch vụ trên Windows
    if os_name == "Windows":
        for s_name in ["MySQL", "MySQL80", "MySQL57", "MariaDB"]:
            try:
                sc_res = subprocess.run(["sc", "query", s_name], capture_output=True, text=True, timeout=2)
                if "RUNNING" in sc_res.stdout:
                    found_info.append(f"Dịch vụ Windows Service '{s_name}' đang HOẠT ĐỘNG (RUNNING)")
                elif "STOPPED" in sc_res.stdout:
                    found_info.append(f"Dịch vụ Windows Service '{s_name}' đã CÀI ĐẶT nhưng đang DỪNG (STOPPED)")
            except Exception:
                pass

        if Path("C:/xampp/mysql/bin/mysqld.exe").exists():
            found_info.append("Tìm thấy bộ cài XAMPP MySQL tại: C:\\xampp")

        if Path("C:/laragon/bin/mysql").exists():
            found_info.append("Tìm thấy bộ cài Laragon MySQL tại: C:\\laragon")

    # D. Kiểm tra dịch vụ trên Linux
    elif os_name == "Linux":
        for svc in ["mysql", "mariadb", "mysqld"]:
            try:
                s_res = subprocess.run(["systemctl", "is-active", svc], capture_output=True, text=True, timeout=2)
                if s_res.returncode == 0 and "active" in s_res.stdout:
                    found_info.append(f"Systemd service '{svc}' đang HOẠT ĐỘNG (Active)")
            except Exception:
                pass

    return found_info


# =======================================================
# 4. NẠP SCHEMA & TẠO TÀI KHOẢN ADMIN MẪU
# =======================================================
def get_mysql_connector():
    try:
        import mysql.connector
        return mysql.connector
    except ImportError:
        print_error("Chưa cài đặt thư viện 'mysql-connector-python'!")
        print_info("Hãy chạy trong môi trường ảo (.venv) hoặc cài đặt: pip install mysql-connector-python")
        return None


def execute_schema_sql(connection, db_name):
    if not SCHEMA_FILE.exists():
        print_error(f"Không tìm thấy file schema tại: {SCHEMA_FILE}")
        return False

    print_step(f"Khởi tạo bảng và nạp dữ liệu mẫu vào database '{db_name}'...")
    cursor = connection.cursor()

    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        sql_content = f.read()

    statements = []
    current_stmt = []

    for line in sql_content.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("--") or not trimmed:
            continue
        current_stmt.append(line)
        if trimmed.endswith(";"):
            stmt_text = "\n".join(current_stmt).strip()
            stmt_text = stmt_text[:-1].strip()
            if stmt_text:
                statements.append(stmt_text)
            current_stmt = []

    executed_count = 0
    for stmt in statements:
        upper_stmt = stmt.upper().strip()
        if upper_stmt.startswith("CREATE DATABASE") or upper_stmt.startswith("USE "):
            continue

        try:
            cursor.execute(stmt)
            executed_count += 1
        except Exception as e:
            if "already exists" not in str(e).lower():
                print_warning(f"Lỗi thực thi lệnh: {e}")

    connection.commit()
    cursor.close()
    print_success(f"Đã thực thi thành công {executed_count} câu lệnh SQL tạo bảng và nạp dữ liệu mẫu!")
    return True


def ensure_admin_account(connection, db_name):
    """Đảm bảo hệ thống có ít nhất một tài khoản Quản trị viên (Admin)."""
    cursor = connection.cursor(dictionary=True)
    cursor.execute(f"SELECT COUNT(*) AS total FROM `{db_name}`.users WHERE role = 'admin';")
    admin_count = cursor.fetchone()["total"]

    if admin_count == 0:
        print_step("Chưa có tài khoản Quản trị viên (Admin). Đang tạo tài khoản mặc định...")
        hashed_pw = generate_password_hash("admin123")
        sql = f"""
            INSERT INTO `{db_name}`.users (username, password, full_name, role)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(sql, ("admin", hashed_pw, "Quản Trị Viên", "admin"))
        connection.commit()
        print_success("Tạo thành công tài khoản Admin mặc định:")
        print_info("   -> Tên đăng nhập : admin")
        print_info("   -> Mật khẩu      : admin123")
    cursor.close()


def ensure_sample_qa_imported(connection, db_name):
    """Nạp thêm câu hỏi phong phú từ data/sample_qa.json nếu chưa có."""
    if not SAMPLE_QA_FILE.exists():
        return

    cursor = connection.cursor(dictionary=True)
    cursor.execute(f"SELECT COUNT(*) AS total FROM `{db_name}`.qa_data;")
    qa_count = cursor.fetchone()["total"]

    if qa_count < 10:
        print_step("Đang đồng bộ thêm câu hỏi mẫu phong phú từ data/sample_qa.json...")
        try:
            with open(SAMPLE_QA_FILE, "r", encoding="utf-8") as f:
                samples = json.load(f)
            insert_sql = f"""
                INSERT INTO `{db_name}`.qa_data (question, answer, intent, tour_id)
                VALUES (%s, %s, %s, %s)
            """
            added = 0
            for item in samples:
                # Kiểm tra tránh trùng lặp
                cursor.execute(f"SELECT id FROM `{db_name}`.qa_data WHERE question = %s", (item["question"],))
                if not cursor.fetchone():
                    cursor.execute(insert_sql, (
                        item["question"],
                        item["answer"],
                        item["intent"],
                        item.get("tour_id")
                    ))
                    added += 1
            connection.commit()
            if added > 0:
                print_success(f"Đã bổ sung thêm {added} câu hỏi mẫu đa dạng vào tập dữ liệu AI!")
        except Exception as e:
            print_warning(f"Lỗi nạp sample_qa.json: {e}")
    cursor.close()


# =======================================================
# 5. LUỒNG THỰC THI CHÍNH
# =======================================================
def check_and_initialize():
    print(f"{Colors.BOLD}======================================================={Colors.RESET}")
    print(f"{Colors.BOLD}      KIỂM TRA VÀ KHỞI TẠO CƠ SỞ DỮ LIỆU TOURAI      {Colors.RESET}")
    print(f"{Colors.BOLD}======================================================={Colors.RESET}")

    env_config = load_env_config()
    db_host = env_config.get("DB_HOST", "127.0.0.1")
    db_port = int(env_config.get("DB_PORT", "3306"))
    db_user = env_config.get("DB_USER", "root")
    db_pass = env_config.get("DB_PASSWORD", "")
    db_name = env_config.get("DB_NAME", "chatbot_tour")

    print_step("Thông tin kết nối cấu hình trong .env:")
    print(f"  • Máy chủ (Host) : {db_host}")
    print(f"  • Cổng (Port)    : {db_port}")
    print(f"  • Tài khoản      : {db_user}")
    print(f"  • Database       : {db_name}")

    # Bước 1: Kiểm tra cổng TCP
    print_step(f"Kiểm tra kết nối cổng TCP {db_host}:{db_port}...")
    port_open = is_port_open(db_host, db_port)

    if not port_open:
        print_error(f"Không thể kết nối đến cổng {db_port} trên {db_host}!")
        print_warning("Dịch vụ MySQL/MariaDB hiện CHƯA ĐƯỢC BẬT hoặc đang chạy trên cổng khác.")

        print_step("Quét hệ thống để phát hiện môi trường MySQL trên máy...")
        sys_info = check_system_environment()
        for item in sys_info:
            print_info(item)

        print(f"\n{Colors.YELLOW}{Colors.BOLD}👉 HƯỚNG DẪN KHỞI ĐỘNG MYSQL:{Colors.RESET}")
        if platform.system() == "Windows":
            print("  1. Nếu dùng Docker (MySQL + DBeaver):")
            print("     -> Mở Docker Desktop và nhấn Start container MySQL, hoặc chạy: docker start mysql-server")
            print("  2. Nếu dùng XAMPP:")
            print("     -> Mở 'XAMPP Control Panel' và nhấn nút 'Start' tại mục MySQL")
            print("  3. Nếu dùng Windows Service:")
            print("     -> Mở CMD (Run as Administrator) và chạy: net start MySQL80 (hoặc net start MySQL)")
            print("  4. Nếu dùng Laragon:")
            print("     -> Mở Laragon và nhấn 'Start All'")
        else:
            print("  1. Nếu dùng Docker (MySQL + DBeaver):")
            print("     -> Chạy lệnh: docker start mysql-server")
            print("     -> Hoặc vào thư mục docker-compose: cd ~/mysql-dbeaver && docker compose up -d")
            print("  2. Nếu dùng dịch vụ Native Linux:")
            print("     -> Chạy lệnh: sudo systemctl start mysql (hoặc sudo systemctl start mariadb)")

        return False

    print_success(f"Cổng {db_port} đang mở và sẵn sàng tiếp nhận kết nối!")

    # Bước 2: Kiểm tra thư viện kết nối
    mysql_conn = get_mysql_connector()
    if not mysql_conn:
        return False

    # Bước 3: Xác thực tài khoản MySQL
    print_step("Xác thực tài khoản và quyền truy cập MySQL...")
    connection = None
    working_password = db_pass

    passwords_to_try = [db_pass]
    fallback_passwords = ["mysecretpassword", "", "root", "123456", "admin", "password", "Duythai1@"]
    for p in fallback_passwords:
        if p not in passwords_to_try:
            passwords_to_try.append(p)

    connect_host = "127.0.0.1" if db_host in ("localhost", "0.0.0.0") else db_host

    for pwd in passwords_to_try:
        try:
            conn = mysql_conn.connect(
                host=connect_host,
                port=db_port,
                user=db_user,
                password=pwd,
                charset="utf8mb4"
            )
            if conn.is_connected():
                connection = conn
                working_password = pwd
                break
        except mysql_conn.Error:
            continue

    if not connection:
        print_error("Không thể xác thực vào MySQL với tài khoản root!")
        print_warning("Đã thử các mật khẩu thông dụng nhưng không khớp.")
        print_info("Vui lòng mở file .env và cập nhật đúng 'DB_PASSWORD' của MySQL trên máy bạn.")
        return False

    print_success(f"Đăng nhập thành công vào MySQL Server (User: '{db_user}')!")

    # Tự động đồng bộ lại file .env nếu mật khẩu thực tế khác
    if working_password != db_pass:
        print_warning(f"Mật khẩu thực tế của MySQL là '{working_password}' (khác với .env).")
        update_env_file("DB_PASSWORD", working_password)
        print_success("Đã tự động cập nhật mật khẩu mới vào file .env!")

    # Bước 4: Kiểm tra database đã tồn tại chưa
    print_step(f"Kiểm tra sự tồn tại của database '{db_name}'...")
    cursor = connection.cursor()
    cursor.execute("SHOW DATABASES;")
    databases = [d[0] for d in cursor.fetchall()]

    db_exists = db_name in databases

    if db_exists:
        print_success(f"Database '{db_name}' ĐÃ TỒN TẠI trên máy chủ MySQL!")

        cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{db_name}';")
        table_count = cursor.fetchone()[0]

        if table_count > 0:
            print_success(f"Database đã có sẵn {table_count} bảng dữ liệu.")
        else:
            print_warning(f"Database '{db_name}' chưa có bảng nào. Tiến hành nạp schema...")
            connection.database = db_name
            execute_schema_sql(connection, db_name)
    else:
        print_warning(f"Database '{db_name}' CHƯA TỒN TẠI trên máy chủ!")
        print_step(f"Đang tự động tạo mới database '{db_name}'...")

        cursor.execute(f"CREATE DATABASE `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        connection.commit()
        print_success(f"Tạo thành công database '{db_name}'!")

        connection.database = db_name
        execute_schema_sql(connection, db_name)

    # Đảm bảo có tài khoản Admin và nạp thêm mẫu câu hỏi phong phú
    ensure_admin_account(connection, db_name)
    ensure_sample_qa_imported(connection, db_name)

    # Thống kê tổng hợp sau khi hoàn tất
    cursor.execute(f"SELECT COUNT(*) FROM `{db_name}`.tours;")
    tour_count = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM `{db_name}`.qa_data;")
    qa_count = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM `{db_name}`.users;")
    user_count = cursor.fetchone()[0]

    cursor.close()
    connection.close()

    print(f"\n{Colors.GREEN}{Colors.BOLD}======================================================={Colors.RESET}")
    print(f"{Colors.GREEN}{Colors.BOLD}   ✔ HOÀN TẤT: CƠ SỞ DỮ LIỆU ĐÃ SẴN SÀNG CHO DỰ ÁN!   {Colors.RESET}")
    print(f"{Colors.GREEN}{Colors.BOLD}======================================================={Colors.RESET}")
    print_info(f"Dữ liệu hiện tại: {tour_count} Tours | {qa_count} Câu hỏi mẫu AI | {user_count} Tài khoản")
    print_info("Tài khoản quản trị viên Admin: [admin / admin123]")
    print_info("Khởi chạy ứng dụng: python app.py")
    return True


if __name__ == "__main__":
    success = check_and_initialize()
    sys.exit(0 if success else 1)
