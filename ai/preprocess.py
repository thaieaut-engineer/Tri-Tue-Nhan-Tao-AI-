import re
import unicodedata


def remove_accents(text):
    """
    Chuyển tiếng Việt có dấu về không dấu.
    Ví dụ:
    'Đà Nẵng' -> 'Da Nang'
    """

    text = unicodedata.normalize("NFD", text)

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )

    return text.replace("đ", "d").replace("Đ", "D")


def preprocess_text(text):
    """
    Chuẩn hóa câu hỏi trước khi đưa vào mô hình AI.
    """

    if not text:
        return ""

    text = text.lower()

    # Chuẩn hóa một số cách viết thường gặp
    text = text.replace("đà nẵng", "da nang")
    text = text.replace("nha trang", "nha trang")
    text = text.replace("hạ long", "ha long")

    # Bỏ dấu tiếng Việt
    text = remove_accents(text)

    # Chỉ giữ chữ, số và khoảng trắng
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    return text