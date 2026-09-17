import re
import unicodedata
import difflib

# BẢNG CHUẨN HÓA TỪ VIẾT TẮT, TEENCODE VÀ TỪ LÓNG THƯỜNG GẶP KHI CHAT
ABBREVIATION_MAP = {
    r"\bks\b": "khach san",
    r"\bksan\b": "khach san",
    r"\bkhasch\b": "khach",
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
    r"\bgtgt\b": "hoa don vat",
    r"\bfanshipan\b": "fansipan",
    r"\bphanxipang\b": "fansipan"
}

# CÁC LỖI CHÍNH TẢ VIẾT LIỀN HOẶC GÕ SAI PHỔ BIẾN
COMMON_TYPO_MAP = {
    r"\bdanang\b": "da nang",
    r"\bda nagn\b": "da nang",
    r"\bnhatrang\b": "nha trang",
    r"\bnha trnag\b": "nha trang",
    r"\bhalong\b": "ha long",
    r"\bha logn\b": "ha long",
    r"\bphuquoc\b": "phu quoc",
    r"\bphu qouc\b": "phu quoc",
    r"\bdalat\b": "da lat",
    r"\bda lta\b": "da lat",
    r"\bsapa\b": "sa pa",
    r"\bquynhon\b": "quy nhon",
    r"\bcantho\b": "can tho",
    r"\bhochiminh\b": "ho chi minh",
    r"\bhanoi\b": "ha noi",
    r"\bkhach sn\b": "khach san",
    r"\bkhasch san\b": "khach san",
    r"\blich trnh\b": "lich trinh",
    r"\bthoi tet\b": "thoi tiet"
}

# TỪ ĐIỂN CÁC TỪ KHÓA CHUYÊN NGÀNH PHỤC VỤ FUZZY SPELL CHECK
DOMAIN_KEYWORDS = [
    "khach", "san", "phu", "quoc", "da", "lat", "ha", "long",
    "nha", "trang", "nang", "sa", "pa", "quy", "nhon", "can", "tho",
    "lich", "trinh", "thoi", "tiet", "fansipan", "vinpearl", "vinwonders"
]


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


def correct_travel_typos(text):
    """
    XỬ LÝ LỖI CHÍNH TẢ & TỪ LÓNG (SPELLING CORRECTION & NORMALIZATION):
    - Khắc phục từ viết liền (danang -> da nang, phuquoc -> phu quoc).
    - Khắc phục từ đảo ký tự khi gõ phím nhanh (phu qouc, da lta, nha trnag, ha logn).
    - Áp dụng thuật toán Levenshtein Distance / Fuzzy matching nếu độ tương đồng >= 0.82.
    """
    if not text:
        return ""

    t_low = text.lower().strip()

    # 1. Khắc phục các cụm từ gõ sai phổ biến
    for pattern, repl in COMMON_TYPO_MAP.items():
        t_low = re.sub(pattern, repl, t_low)

    # 2. Thay thế từ viết tắt / từ lóng
    for pattern, repl in ABBREVIATION_MAP.items():
        t_low = re.sub(pattern, repl, t_low)

    # 3. Fuzzy matching cho từng từ đơn (nếu từ không chuẩn và có độ tương đồng cao với từ điển)
    words = t_low.split()
    corrected_words = []
    for w in words:
        clean_w = remove_accents(w)
        if len(clean_w) >= 3 and clean_w not in DOMAIN_KEYWORDS and not clean_w.isdigit():
            matches = difflib.get_close_matches(clean_w, DOMAIN_KEYWORDS, n=1, cutoff=0.82)
            if matches:
                corrected_words.append(matches[0])
            else:
                corrected_words.append(w)
        else:
            corrected_words.append(w)

    return " ".join(corrected_words)


def preprocess_text(text):
    """
    CHUẨN HÓA TIỀN XỬ LÝ VĂN BẢN TOÀN DIỆN CHO MÔ HÌNH AI:
    1. Sửa lỗi chính tả & từ lóng (Typo Correction).
    2. Chuẩn hóa địa danh.
    3. Bỏ dấu tiếng Việt chuẩn hóa.
    4. Chỉ giữ chữ, số và khoảng trắng.
    """
    if not text:
        return ""

    # Bước 1: Sửa lỗi chính tả và chuẩn hóa từ lóng
    text = correct_travel_typos(text)

    # Bước 2: Bỏ dấu tiếng Việt
    text = remove_accents(text)

    # Bước 3: Chỉ giữ chữ, số và khoảng trắng
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Bước 4: Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    return text