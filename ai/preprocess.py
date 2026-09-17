import re
import unicodedata

# BẢNG CHUẨN HÓA TỪ VIẾT TẮT, TEENCODE VÀ TỪ LÓNG THƯỜNG GẶP KHI CHAT
ABBREVIATION_MAP = {
    r"\bks\b": "khach san",
    r"\bksan\b": "khach san",
    r"\bbn\b": "bao nhieu",
    r"\bbao nhiu\b": "bao nhieu",
    r"\bbnhieu\b": "bao nhieu",
    r"\bbnh\b": "bao nhieu",
    r"\bbao tien\b": "bao nhieu tien",
    r"\bk\b": "khong",
    r"\bko\b": "khong",
    r"\bhok\b": "khong",
    r"\bkh\b": "khong",
    r"\bdc\b": "duoc",
    r"\bđc\b": "duoc",
    r"\bntn\b": "nhu the nao",
    r"\bthe nao\b": "nhu the nao",
    r"\bve mb\b": "ve may bay",
    r"\bvmb\b": "ve may bay",
    r"\boto\b": "o to",
    r"\bxe hoi\b": "o to",
    r"\bhdv\b": "huong dan vien",
    r"\btphcm\b": "ho chi minh",
    r"\bhcm\b": "ho chi minh",
    r"\bsg\b": "sai gon",
    r"\bhn\b": "ha noi",
    r"\bpq\b": "phu quoc",
    r"\bđl\b": "da lat",
    r"\bdl\b": "da lat",
    r"\bđn\b": "da nang",
    r"\bdn\b": "da nang",
    r"\bnt\b": "nha trang",
    r"\bhl\b": "ha long",
    r"\bqn\b": "quy nhon",
    r"\bct\b": "can tho",
    r"\bsp\b": "sa pa",
    r"\bsapa\b": "sa pa",
    r"\btt\b": "thoi tiet",
    r"\bvat\b": "hoa don vat",
    r"\bgtgt\b": "hoa don vat"
}


def remove_accents(text):
    """
    Chuyển tiếng Việt có dấu về không dấu chuẩn Unicode.
    Ví dụ: 'Đà Nẵng' -> 'Da Nang'
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )

    return text.replace("đ", "d").replace("Đ", "D")


def preprocess_text(text):
    """
    CHUẨN HÓA TIỀN XỬ LÝ VĂN BẢN NÂNG CAO CHO MÔ HÌNH AI:
    1. Chuyển về chữ thường (lowercase).
    2. Chuẩn hóa từ viết tắt, từ lóng tiếng Việt (Abbreviations & Slang Normalization).
    3. Chuẩn hóa tên các địa danh nổi bật.
    4. Bỏ dấu tiếng Việt phục vụ trích xuất đặc trưng thống nhất.
    5. Loại bỏ ký tự đặc biệt, giữ lại chữ và số.
    6. Xóa khoảng trắng thừa.
    """
    if not text:
        return ""

    text = text.lower().strip()

    # Chuẩn hóa từ viết tắt theo biểu thức chính quy
    for pattern, replacement in ABBREVIATION_MAP.items():
        text = re.sub(pattern, replacement, text)

    # Chuẩn hóa tên địa danh trọng điểm
    text = text.replace("đà nẵng", "da nang")
    text = text.replace("nha trang", "nha trang")
    text = text.replace("hạ long", "ha long")
    text = text.replace("phú quốc", "phu quoc")
    text = text.replace("đà lạt", "da lat")
    text = text.replace("sa pa", "sa pa")
    text = text.replace("quy nhơn", "quy nhon")
    text = text.replace("cần thơ", "can tho")

    # Bỏ dấu tiếng Việt
    text = remove_accents(text)

    # Chỉ giữ chữ, số và khoảng trắng
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    return text