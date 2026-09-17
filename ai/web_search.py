"""
ai/web_search.py
Module tìm kiếm thông tin du lịch thông minh trên mạng Internet:
- Khử nhập nhằng thực thể (Entity Disambiguation) tránh tìm nhầm tiểu sử nhân vật.
- Định hướng từ khóa chuyên ngành du lịch (Travel Query Reformulation).
- Ưu tiên DuckDuckGo Search (DDGS) với cơ chế lọc kết quả sát câu hỏi.
- Dự phòng Wikipedia tiếng Việt có bộ lọc loại trừ bài viết tiểu sử cá nhân.
"""

import re
import urllib.parse
import requests

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False


def clean_html(raw_html):
    """Loại bỏ thẻ HTML thừa nếu có."""
    clean_text = re.sub(r"<.*?>", "", raw_html)
    return re.sub(r"\s+", " ", clean_text).strip()


def clean_snippet(text, max_len=280):
    """Làm sạch văn bản tóm tắt, cắt bỏ các đoạn rác từ máy tìm kiếm."""
    if not text:
        return ""
    text = clean_html(text)
    text = re.sub(r"See full list on [^\s]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s*·", "", text)
    text = re.sub(r"\d+\s+(hours?|days?|minutes?)\s+ago\s*·", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "..."
    return text


def reformulate_travel_query(query):
    """
    ĐỊNH HƯỚNG TỪ KHÓA TÌM KIẾM DU LỊCH (QUERY REFORMULATION):
    - Khử nhập nhằng địa danh (ví dụ 'Hồ Chí Minh' -> 'TP Hồ Chí Minh Sài Gòn').
    - Bổ sung ngữ cảnh du lịch để máy tìm kiếm trả về danh thắng, khách sạn, ẩm thực.
    """
    q = query.strip()
    q_low = q.lower()

    # 1. Khử nhập nhằng địa danh Hồ Chí Minh
    if any(k in q_low for k in ["hồ chí minh", "ho chi minh", "hcm", "tphcm"]):
        q = re.sub(r"\bhồ chí minh\b", "TP Hồ Chí Minh", q, flags=re.IGNORECASE)
        q = re.sub(r"\bho chi minh\b", "TP Hồ Chí Minh", q, flags=re.IGNORECASE)
        q = re.sub(r"\btphcm\b", "TP Hồ Chí Minh", q, flags=re.IGNORECASE)
        q = re.sub(r"\bhcm\b", "TP Hồ Chí Minh", q, flags=re.IGNORECASE)
        if "sài gòn" not in q_low and "sai gon" not in q_low:
            q += " Sài Gòn"

    # 2. Bổ sung từ khóa du lịch theo ý định
    if any(w in q_low for w in ["điểm tham quan", "chơi gì", "có gì đẹp", "đi đâu", "tham quan", "check in"]):
        if "du lịch" not in q_low:
            q = f"địa điểm du lịch tham quan {q}"
    elif any(w in q_low for w in ["khách sạn", "resort", "homestay", "chỗ ở", "nhà nghỉ"]):
        if "đẹp" not in q_low and "tốt" not in q_low:
            q = f"{q} tốt nhất"
    elif any(w in q_low for w in ["ăn gì", "đặc sản", "món ngon", "quán ngon", "ẩm thực"]):
        if "ẩm thực" not in q_low:
            q = f"món ngon đặc sản ẩm thực {q}"

    return q


def is_irrelevant_biography(title, extract):
    """
    BỘ LỌC CHỐNG LỆCH CHỦ ĐỀ:
    Phát hiện và loại trừ bài viết tiểu sử nhân vật chính trị/lịch sử
    khi người dùng đang tìm kiếm thông tin du lịch, địa điểm.
    """
    text = (title + " " + extract).lower()
    bio_flags = [
        "tên khai sinh", "nhà cách mạng", "chính khách", "chủ tịch nước",
        "tổng bí thư", "thủ tướng", "chính trị gia", "nhà văn", "nhà thơ",
        "sinh ngày", "mất ngày", "sinh năm", "mất năm", "từng là chủ tịch"
    ]
    return any(flag in text for flag in bio_flags)


def search_with_ddgs(query, max_results=3):
    """Tìm kiếm bằng thư viện DDGS (DuckDuckGo)."""
    if not DDGS_AVAILABLE:
        return []

    results = []
    try:
        search_query = query.strip()
        with DDGS(timeout=8) as ddgs:
            raw_results = list(ddgs.text(search_query, max_results=max_results))
            for item in raw_results:
                title = item.get("title", "")
                snippet = clean_snippet(item.get("body", ""))
                url = item.get("href", "")
                if title and snippet:
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "source": "DuckDuckGo"
                    })
    except Exception as e:
        print("Lỗi tìm kiếm DDGS:", e)

    return results


def search_with_wikipedia(query):
    """Tìm kiếm tóm tắt trên Wikipedia tiếng Việt có kiểm tra tính liên quan du lịch."""
    results = []
    try:
        # Nếu câu hỏi về TP Hồ Chí Minh, chuyển hướng cụ thể sang bài viết thành phố
        search_term = query
        if any(k in query.lower() for k in ["hồ chí minh", "ho chi minh", "hcm", "tphcm"]):
            search_term = "Thành phố Hồ Chí Minh"

        search_url = "https://vi.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": search_term,
            "utf8": 1,
            "format": "json"
        }
        headers = {"User-Agent": "TourAI-Bot/1.0 (travel assistant)"}
        resp = requests.get(search_url, params=params, headers=headers, timeout=5).json()
        search_items = resp.get("query", {}).get("search", [])

        if search_items:
            for top_item in search_items[:2]:
                title = top_item["title"]

                summary_url = f"https://vi.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                sum_resp = requests.get(summary_url, headers=headers, timeout=5)
                if sum_resp.status_code == 200:
                    sum_data = sum_resp.json()
                    extract = clean_snippet(sum_data.get("extract", ""), max_len=300)
                    page_url = sum_data.get("content_urls", {}).get("desktop", {}).get("page", "")

                    # Kiểm tra loại trừ bài viết tiểu sử nhân vật không liên quan du lịch
                    if is_irrelevant_biography(title, extract):
                        continue

                    if extract:
                        results.append({
                            "title": title,
                            "snippet": extract,
                            "url": page_url or f"https://vi.wikipedia.org/wiki/{urllib.parse.quote(title)}",
                            "source": "Wikipedia tiếng Việt"
                        })
                        return results

    except Exception as e:
        print("Lỗi tìm kiếm Wikipedia:", e)

    return results


def search_web_for_travel(query, max_results=3):
    """
    Hàm tổng hợp tìm kiếm trên mạng:
    1. Định hướng từ khóa du lịch (Reformulation).
    2. Ưu tiên DuckDuckGo Search.
    3. Dự phòng Wikipedia tiếng Việt có bộ lọc chống lệch chủ đề.
    """
    reformulated = reformulate_travel_query(query)
    results = search_with_ddgs(reformulated, max_results=max_results)

    # Nếu truy vấn định hướng chưa có kết quả, thử truy vấn nguyên bản
    if not results and query != reformulated:
        results = search_with_ddgs(query, max_results=max_results)

    # Dự phòng Wikipedia nếu DuckDuckGo bị chặn kết nối
    if not results:
        results = search_with_wikipedia(reformulated)

    return results


def format_web_response(query, results):
    """
    Định dạng kết quả tìm kiếm trên mạng thành câu trả lời tự nhiên, lịch sự và dễ đọc.
    """
    if not results:
        return (
            f"🔍 Tôi đã thử tìm kiếm trên mạng về '{query}' nhưng chưa tìm thấy thông tin phù hợp. "
            f"Bạn có thể thử hỏi chi tiết hơn hoặc tham khảo các tour du lịch hiện có nhé!"
        )

    response_lines = [
        f"🌐 Thông tin du lịch tìm kiếm trực tuyến về \"{query}\":\n"
    ]

    for idx, r in enumerate(results[:3], start=1):
        title = r["title"]
        snippet = r["snippet"].strip()
        url = r.get("url", "")

        response_lines.append(f"📌 **{idx}. {title}**")
        response_lines.append(f"{snippet}")
        if url:
            response_lines.append(f"🔗 Xem thêm: {url}")
        response_lines.append("")

    response_lines.append("💡 Nếu bạn muốn tham khảo các tour du lịch trọn gói trong hệ thống (Đà Nẵng, Nha Trang, Hạ Long, Phú Quốc, Đà Lạt, Sa Pa, Quy Nhơn, Cần Thơ), hãy hỏi tôi nhé!")

    return "\n".join(response_lines).strip()
