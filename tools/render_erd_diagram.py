#!/usr/bin/env python3
"""
render_erd_diagram.py
Tạo ảnh Sơ đồ Thực thể Quan hệ (ERD) chất lượng cao (300 DPI) cho CSDL chatbot_tour (10 bảng chuẩn hóa).
"""

import os
import subprocess
from pathlib import Path

ARTIFACT_DIR = Path("/home/kennysk/.gemini/antigravity-cli/brain/49330d9f-2d3d-407d-816d-eccdb215f3a2")
REPORT_IMAGES_DIR = Path("/home/kennysk/Tri-Tue-Nhan-Tao-AI-/bao_cao/report_images")

REPORT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

out_artifact_png = ARTIFACT_DIR / "erd_chatbot_tour.png"
out_report_png = REPORT_IMAGES_DIR / "fig_2_2_erd.png"

dot_content = """digraph ERD {
    graph [
        rankdir="TB",
        nodesep=0.7,
        ranksep=1.0,
        bgcolor="#F8FAFC",
        splines="spline",
        fontname="DejaVu Sans, Arial, Helvetica",
        pad="0.5"
    ];
    node [
        shape="none",
        fontname="DejaVu Sans, Arial, Helvetica",
        fontsize=11
    ];
    edge [
        fontname="DejaVu Sans, Arial, Helvetica",
        fontsize=9,
        penwidth=1.6
    ];

    // TITLE NODE
    title [
        label=<
            <table border="0" cellborder="0" cellspacing="0" cellpadding="8">
                <tr>
                    <td align="center"><font point-size="18" color="#0F172A"><b>SƠ ĐỒ THỰC THỂ QUAN HỆ (ERD) - CƠ SỞ DỮ LIỆU TOURAI</b></font></td>
                </tr>
                <tr>
                    <td align="center"><font point-size="12" color="#475569">Hệ Thống Quản Lý &amp; Tư Vấn Tour Du Lịch Thông Minh (10 Bảng Chuẩn Hóa MySQL)</font></td>
                </tr>
            </table>
        >
    ];

    // ==========================================
    // 1. NHÓM TÀI KHOẢN & NGƯỜI DÙNG
    // ==========================================
    users [
        label=<
            <table border="1" cellborder="0" cellspacing="0" cellpadding="5" bgcolor="#FFFFFF" color="#2563EB" style="rounded">
                <tr><td colspan="3" bgcolor="#1D4ED8" align="center"><font color="#FFFFFF"><b>users (Người dùng &amp; Admin)</b></font></td></tr>
                <tr bgcolor="#F1F5F9"><td align="left"><b>Cột dữ liệu</b></td><td align="center"><b>Khóa</b></td><td align="right"><b>Kiểu dữ liệu</b></td></tr>
                <tr><td align="left">id</td><td align="center"><font color="#DC2626"><b>[PK]</b></font></td><td align="right"><font color="#64748B">INT AUTO_INC</font></td></tr>
                <tr><td align="left">username</td><td align="center"><font color="#2563EB">[UQ]</font></td><td align="right"><font color="#64748B">VARCHAR(50)</font></td></tr>
                <tr><td align="left">password</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(255)</font></td></tr>
                <tr><td align="left">full_name</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(100)</font></td></tr>
                <tr><td align="left">role</td><td align="center"></td><td align="right"><font color="#64748B">ENUM('user','admin')</font></td></tr>
                <tr><td align="left">email</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(100)</font></td></tr>
                <tr><td align="left">phone</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(20)</font></td></tr>
                <tr><td align="left">created_at</td><td align="center"></td><td align="right"><font color="#64748B">TIMESTAMP</font></td></tr>
            </table>
        >
    ];

    // ==========================================
    // 2. NHÓM DANH MỤC & TOUR DU LỊCH
    // ==========================================
    categories [
        label=<
            <table border="1" cellborder="0" cellspacing="0" cellpadding="5" bgcolor="#FFFFFF" color="#0D9488" style="rounded">
                <tr><td colspan="3" bgcolor="#0F766E" align="center"><font color="#FFFFFF"><b>categories (Danh mục Tour)</b></font></td></tr>
                <tr bgcolor="#F1F5F9"><td align="left"><b>Cột dữ liệu</b></td><td align="center"><b>Khóa</b></td><td align="right"><b>Kiểu dữ liệu</b></td></tr>
                <tr><td align="left">id</td><td align="center"><font color="#DC2626"><b>[PK]</b></font></td><td align="right"><font color="#64748B">INT AUTO_INC</font></td></tr>
                <tr><td align="left">name</td><td align="center"><font color="#2563EB">[UQ]</font></td><td align="right"><font color="#64748B">VARCHAR(100)</font></td></tr>
                <tr><td align="left">description</td><td align="center"></td><td align="right"><font color="#64748B">TEXT</font></td></tr>
                <tr><td align="left">created_at</td><td align="center"></td><td align="right"><font color="#64748B">TIMESTAMP</font></td></tr>
            </table>
        >
    ];

    tours [
        label=<
            <table border="1" cellborder="0" cellspacing="0" cellpadding="5" bgcolor="#FFFFFF" color="#2563EB" style="rounded">
                <tr><td colspan="3" bgcolor="#2563EB" align="center"><font color="#FFFFFF"><b>tours (20 Tour du lịch)</b></font></td></tr>
                <tr bgcolor="#F1F5F9"><td align="left"><b>Cột dữ liệu</b></td><td align="center"><b>Khóa</b></td><td align="right"><b>Kiểu dữ liệu</b></td></tr>
                <tr><td align="left">id</td><td align="center"><font color="#DC2626"><b>[PK]</b></font></td><td align="right"><font color="#64748B">INT AUTO_INC</font></td></tr>
                <tr><td align="left">category_id</td><td align="center"><font color="#16A34A"><b>[FK]</b></font></td><td align="right"><font color="#64748B">INT (Nullable)</font></td></tr>
                <tr><td align="left">name</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(200)</font></td></tr>
                <tr><td align="left">destination</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(200)</font></td></tr>
                <tr><td align="left">duration</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(50)</font></td></tr>
                <tr><td align="left">price</td><td align="center"></td><td align="right"><font color="#64748B">DECIMAL(12,2)</font></td></tr>
                <tr><td align="left">description</td><td align="center"></td><td align="right"><font color="#64748B">TEXT</font></td></tr>
                <tr><td align="left">image</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(255)</font></td></tr>
                <tr><td align="left">created_at</td><td align="center"></td><td align="right"><font color="#64748B">TIMESTAMP</font></td></tr>
            </table>
        >
    ];

    tour_schedule [
        label=<
            <table border="1" cellborder="0" cellspacing="0" cellpadding="5" bgcolor="#FFFFFF" color="#059669" style="rounded">
                <tr><td colspan="3" bgcolor="#059669" align="center"><font color="#FFFFFF"><b>tour_schedule (52 Lịch trình ngày)</b></font></td></tr>
                <tr bgcolor="#F1F5F9"><td align="left"><b>Cột dữ liệu</b></td><td align="center"><b>Khóa</b></td><td align="right"><b>Kiểu dữ liệu</b></td></tr>
                <tr><td align="left">id</td><td align="center"><font color="#DC2626"><b>[PK]</b></font></td><td align="right"><font color="#64748B">INT AUTO_INC</font></td></tr>
                <tr><td align="left">tour_id</td><td align="center"><font color="#16A34A"><b>[FK]</b></font></td><td align="right"><font color="#64748B">INT (Cascade)</font></td></tr>
                <tr><td align="left">day_number</td><td align="center"></td><td align="right"><font color="#64748B">INT</font></td></tr>
                <tr><td align="left">location</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(200)</font></td></tr>
                <tr><td align="left">activity</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(255)</font></td></tr>
                <tr><td align="left">description</td><td align="center"></td><td align="right"><font color="#64748B">TEXT</font></td></tr>
            </table>
        >
    ];

    // ==========================================
    // 3. NHÓM TRI THỨC AI & CHATBOT
    // ==========================================
    qa_data [
        label=<
            <table border="1" cellborder="0" cellspacing="0" cellpadding="5" bgcolor="#FFFFFF" color="#7C3AED" style="rounded">
                <tr><td colspan="3" bgcolor="#6D28D9" align="center"><font color="#FFFFFF"><b>qa_data (1.290 Tri thức AI)</b></font></td></tr>
                <tr bgcolor="#F1F5F9"><td align="left"><b>Cột dữ liệu</b></td><td align="center"><b>Khóa</b></td><td align="right"><b>Kiểu dữ liệu</b></td></tr>
                <tr><td align="left">id</td><td align="center"><font color="#DC2626"><b>[PK]</b></font></td><td align="right"><font color="#64748B">INT AUTO_INC</font></td></tr>
                <tr><td align="left">tour_id</td><td align="center"><font color="#16A34A"><b>[FK]</b></font></td><td align="right"><font color="#64748B">INT (Nullable)</font></td></tr>
                <tr><td align="left">question</td><td align="center"></td><td align="right"><font color="#64748B">TEXT</font></td></tr>
                <tr><td align="left">answer</td><td align="center"></td><td align="right"><font color="#64748B">TEXT</font></td></tr>
                <tr><td align="left">intent</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(100)</font></td></tr>
                <tr><td align="left">created_at</td><td align="center"></td><td align="right"><font color="#64748B">TIMESTAMP</font></td></tr>
            </table>
        >
    ];

    chat_sessions [
        label=<
            <table border="1" cellborder="0" cellspacing="0" cellpadding="5" bgcolor="#FFFFFF" color="#EA580C" style="rounded">
                <tr><td colspan="3" bgcolor="#C2410C" align="center"><font color="#FFFFFF"><b>chat_sessions (Phiên hội thoại)</b></font></td></tr>
                <tr bgcolor="#F1F5F9"><td align="left"><b>Cột dữ liệu</b></td><td align="center"><b>Khóa</b></td><td align="right"><b>Kiểu dữ liệu</b></td></tr>
                <tr><td align="left">id</td><td align="center"><font color="#DC2626"><b>[PK]</b></font></td><td align="right"><font color="#64748B">INT AUTO_INC</font></td></tr>
                <tr><td align="left">user_id</td><td align="center"><font color="#16A34A"><b>[FK]</b></font></td><td align="right"><font color="#64748B">INT (Cascade)</font></td></tr>
                <tr><td align="left">title</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(200)</font></td></tr>
                <tr><td align="left">created_at</td><td align="center"></td><td align="right"><font color="#64748B">TIMESTAMP</font></td></tr>
                <tr><td align="left">updated_at</td><td align="center"></td><td align="right"><font color="#64748B">TIMESTAMP</font></td></tr>
            </table>
        >
    ];

    chat_history [
        label=<
            <table border="1" cellborder="0" cellspacing="0" cellpadding="5" bgcolor="#FFFFFF" color="#EA580C" style="rounded">
                <tr><td colspan="3" bgcolor="#EA580C" align="center"><font color="#FFFFFF"><b>chat_history (Lịch sử &amp; AI Tự học)</b></font></td></tr>
                <tr bgcolor="#F1F5F9"><td align="left"><b>Cột dữ liệu</b></td><td align="center"><b>Khóa</b></td><td align="right"><b>Kiểu dữ liệu</b></td></tr>
                <tr><td align="left">id</td><td align="center"><font color="#DC2626"><b>[PK]</b></font></td><td align="right"><font color="#64748B">INT AUTO_INC</font></td></tr>
                <tr><td align="left">session_id</td><td align="center"><font color="#16A34A"><b>[FK]</b></font></td><td align="right"><font color="#64748B">INT (Cascade)</font></td></tr>
                <tr><td align="left">question</td><td align="center"></td><td align="right"><font color="#64748B">TEXT</font></td></tr>
                <tr><td align="left">answer</td><td align="center"></td><td align="right"><font color="#64748B">TEXT</font></td></tr>
                <tr><td align="left">intent</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(100)</font></td></tr>
                <tr><td align="left">confidence</td><td align="center"></td><td align="right"><font color="#64748B">FLOAT</font></td></tr>
                <tr><td align="left">is_learned</td><td align="center"></td><td align="right"><font color="#64748B">TINYINT(1)</font></td></tr>
                <tr><td align="left">created_at</td><td align="center"></td><td align="right"><font color="#64748B">TIMESTAMP</font></td></tr>
            </table>
        >
    ];

    // ==========================================
    // 4. NHÓM NGHIỆP VỤ & TƯƠNG TÁC
    // ==========================================
    bookings [
        label=<
            <table border="1" cellborder="0" cellspacing="0" cellpadding="5" bgcolor="#FFFFFF" color="#0284C7" style="rounded">
                <tr><td colspan="3" bgcolor="#0369A1" align="center"><font color="#FFFFFF"><b>bookings (Đơn đặt Tour)</b></font></td></tr>
                <tr bgcolor="#F1F5F9"><td align="left"><b>Cột dữ liệu</b></td><td align="center"><b>Khóa</b></td><td align="right"><b>Kiểu dữ liệu</b></td></tr>
                <tr><td align="left">id</td><td align="center"><font color="#DC2626"><b>[PK]</b></font></td><td align="right"><font color="#64748B">INT AUTO_INC</font></td></tr>
                <tr><td align="left">user_id</td><td align="center"><font color="#16A34A"><b>[FK]</b></font></td><td align="right"><font color="#64748B">INT</font></td></tr>
                <tr><td align="left">tour_id</td><td align="center"><font color="#16A34A"><b>[FK]</b></font></td><td align="right"><font color="#64748B">INT</font></td></tr>
                <tr><td align="left">full_name</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(150)</font></td></tr>
                <tr><td align="left">phone</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(25)</font></td></tr>
                <tr><td align="left">email</td><td align="center"></td><td align="right"><font color="#64748B">VARCHAR(150)</font></td></tr>
                <tr><td align="left">start_date</td><td align="center"></td><td align="right"><font color="#64748B">DATE</font></td></tr>
                <tr><td align="left">adults</td><td align="center"></td><td align="right"><font color="#64748B">INT</font></td></tr>
                <tr><td align="left">children</td><td align="center"></td><td align="right"><font color="#64748B">INT</font></td></tr>
                <tr><td align="left">total_price</td><td align="center"></td><td align="right"><font color="#64748B">DECIMAL(12,2)</font></td></tr>
                <tr><td align="left">status</td><td align="center"></td><td align="right"><font color="#64748B">ENUM('pending',...)</font></td></tr>
                <tr><td align="left">note</td><td align="center"></td><td align="right"><font color="#64748B">TEXT</font></td></tr>
                <tr><td align="left">created_at</td><td align="center"></td><td align="right"><font color="#64748B">TIMESTAMP</font></td></tr>
            </table>
        >
    ];

    reviews [
        label=<
            <table border="1" cellborder="0" cellspacing="0" cellpadding="5" bgcolor="#FFFFFF" color="#D97706" style="rounded">
                <tr><td colspan="3" bgcolor="#B45309" align="center"><font color="#FFFFFF"><b>reviews (Đánh giá 5 sao)</b></font></td></tr>
                <tr bgcolor="#F1F5F9"><td align="left"><b>Cột dữ liệu</b></td><td align="center"><b>Khóa</b></td><td align="right"><b>Kiểu dữ liệu</b></td></tr>
                <tr><td align="left">id</td><td align="center"><font color="#DC2626"><b>[PK]</b></font></td><td align="right"><font color="#64748B">INT AUTO_INC</font></td></tr>
                <tr><td align="left">user_id</td><td align="center"><font color="#16A34A"><b>[FK]</b></font></td><td align="right"><font color="#64748B">INT</font></td></tr>
                <tr><td align="left">tour_id</td><td align="center"><font color="#16A34A"><b>[FK]</b></font></td><td align="right"><font color="#64748B">INT</font></td></tr>
                <tr><td align="left">rating</td><td align="center"></td><td align="right"><font color="#64748B">INT (1-5)</font></td></tr>
                <tr><td align="left">comment</td><td align="center"></td><td align="right"><font color="#64748B">TEXT</font></td></tr>
                <tr><td align="left">created_at</td><td align="center"></td><td align="right"><font color="#64748B">TIMESTAMP</font></td></tr>
            </table>
        >
    ];

    favorites [
        label=<
            <table border="1" cellborder="0" cellspacing="0" cellpadding="5" bgcolor="#FFFFFF" color="#DB2777" style="rounded">
                <tr><td colspan="3" bgcolor="#BE185D" align="center"><font color="#FFFFFF"><b>favorites (Tour yêu thích)</b></font></td></tr>
                <tr bgcolor="#F1F5F9"><td align="left"><b>Cột dữ liệu</b></td><td align="center"><b>Khóa</b></td><td align="right"><b>Kiểu dữ liệu</b></td></tr>
                <tr><td align="left">id</td><td align="center"><font color="#DC2626"><b>[PK]</b></font></td><td align="right"><font color="#64748B">INT AUTO_INC</font></td></tr>
                <tr><td align="left">user_id</td><td align="center"><font color="#16A34A"><b>[FK]</b></font></td><td align="right"><font color="#64748B">INT</font></td></tr>
                <tr><td align="left">tour_id</td><td align="center"><font color="#16A34A"><b>[FK]</b></font></td><td align="right"><font color="#64748B">INT</font></td></tr>
                <tr><td align="left">created_at</td><td align="center"></td><td align="right"><font color="#64748B">TIMESTAMP</font></td></tr>
            </table>
        >
    ];

    // ==========================================
    // CÁC QUAN HỆ KHÓA NGOẠI (FOREIGN KEYS)
    // ==========================================
    title -> users [style=invis];

    // Category -> Tours
    categories -> tours [
        color="#0D9488",
        label=" 1 : N (category_id)",
        arrowhead="crow",
        arrowtail="none",
        dir="both"
    ];

    // Tours -> Tour Schedule
    tours -> tour_schedule [
        color="#059669",
        label=" 1 : N (tour_id)",
        arrowhead="crow",
        arrowtail="none",
        dir="both"
    ];

    // Tours -> QA Data
    tours -> qa_data [
        color="#7C3AED",
        label=" 0..1 : N (tour_id)",
        style="dashed",
        arrowhead="crow",
        arrowtail="none",
        dir="both"
    ];

    // Users -> Chat Sessions -> Chat History
    users -> chat_sessions [
        color="#C2410C",
        label=" 1 : N (user_id)",
        arrowhead="crow",
        arrowtail="none",
        dir="both"
    ];
    chat_sessions -> chat_history [
        color="#EA580C",
        label=" 1 : N (session_id)",
        arrowhead="crow",
        arrowtail="none",
        dir="both"
    ];

    // Users & Tours -> Bookings
    users -> bookings [
        color="#0284C7",
        label=" 1 : N (user_id)",
        arrowhead="crow",
        arrowtail="none",
        dir="both"
    ];
    tours -> bookings [
        color="#0284C7",
        label=" 1 : N (tour_id)",
        arrowhead="crow",
        arrowtail="none",
        dir="both"
    ];

    // Users & Tours -> Reviews
    users -> reviews [
        color="#D97706",
        label=" 1 : N (user_id)",
        arrowhead="crow",
        arrowtail="none",
        dir="both"
    ];
    tours -> reviews [
        color="#D97706",
        label=" 1 : N (tour_id)",
        arrowhead="crow",
        arrowtail="none",
        dir="both"
    ];

    // Users & Tours -> Favorites
    users -> favorites [
        color="#DB2777",
        label=" 1 : N (user_id)",
        arrowhead="crow",
        arrowtail="none",
        dir="both"
    ];
    tours -> favorites [
        color="#DB2777",
        label=" 1 : N (tour_id)",
        arrowhead="crow",
        arrowtail="none",
        dir="both"
    ];
}
"""

def generate():
    dot_file = REPORT_IMAGES_DIR / "erd_chatbot_tour.dot"
    with open(dot_file, "w", encoding="utf-8") as f:
        f.write(dot_content)
    
    print(f"[*] Đang vẽ ERD sang {out_report_png} (300 DPI)...")
    res1 = subprocess.run(["dot", "-Tpng", "-Gdpi=250", str(dot_file), "-o", str(out_report_png)], capture_output=True)
    if res1.returncode != 0:
        print("Lỗi dot report:", res1.stderr.decode('utf-8'))
    else:
        print(f"✔ Đã tạo thành công: {out_report_png}")

    print(f"[*] Đang sao chép ERD sang Artifact: {out_artifact_png}...")
    res2 = subprocess.run(["dot", "-Tpng", "-Gdpi=250", str(dot_file), "-o", str(out_artifact_png)], capture_output=True)
    if res2.returncode != 0:
        print("Lỗi dot artifact:", res2.stderr.decode('utf-8'))
    else:
        print(f"✔ Đã tạo thành công: {out_artifact_png}")

if __name__ == "__main__":
    generate()
