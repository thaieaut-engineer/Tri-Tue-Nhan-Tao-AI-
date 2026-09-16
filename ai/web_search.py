"""
ai/web_search.py
Module tìm kiếm thông tin du lịch trên mạng Internet khi dữ liệu nội bộ chưa có câu trả lời.
Hỗ trợ:
 - DDGS (DuckDuckGo Search)
 - Wikipedia tiếng Việt API
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
    """Loại bỏ thẻ HTML thừa nếu có"""
    clean_text = re.sub(r"<.*?>", "", raw_html)
    return re.sub(r"\s+", " ", clean_text).strip()


def clean_snippet(text, max_len=280):
    """Làm sạch văn bản tóm tắt, cắt bỏ các đoạn gợi ý lặp và giới hạn độ dài vừa mắt"""
    if not text:
        return ""
    text = clean_html(text)
    # Xóa các cụm thừa từ máy tìm kiếm
    text = re.sub(r"See full list on [^\s]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s*·", "", text)
    text = re.sub(r"\d+\s+(hours?|days?|minutes?)\s+ago\s*·", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "..."
    return text


def search_with_ddgs(query, max_results=3):
    """Tìm kiếm bằng thư viện DDGS (DuckDuckGo)"""
    if not DDGS_AVAILABLE:
        return []

    results = []
    try:
        search_query = query.strip()
        with DDGS(timeout=6) as ddgs:
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
    """Tìm kiếm tóm tắt trên Wikipedia tiếng Việt"""
    results = []
    try:
        search_url = "https://vi.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "utf8": 1,
            "format": "json"
        }
        headers = {"User-Agent": "TourAI-Bot/1.0 (travel assistant)"}
        resp = requests.get(search_url, params=params, headers=headers, timeout=4).json()
        search_items = resp.get("query", {}).get("search", [])

        if search_items:
            top_item = search_items[0]
            title = top_item["title"]

            summary_url = f"https://vi.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
            sum_resp = requests.get(summary_url, headers=headers, timeout=4)
            if sum_resp.status_code == 200:
                sum_data = sum_resp.json()
                extract = clean_snippet(sum_data.get("extract", ""), max_len=300)
                page_url = sum_data.get("content_urls", {}).get("desktop", {}).get("page", "")
                if extract:
                    results.append({
                        "title": title,
                        "snippet": extract,
                        "url": page_url or f"https://vi.wikipedia.org/wiki/{urllib.parse.quote(title)}",
                        "source": "Wikipedia tiếng Việt"
                    })
                    return results

            snippet = clean_snippet(top_item.get("snippet", ""), max_len=200)
            if snippet:
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": f"https://vi.wikipedia.org/wiki/{urllib.parse.quote(title)}",
                    "source": "Wikipedia tiếng Việt"
                })
    except Exception as e:
        print("Lỗi tìm kiếm Wikipedia:", e)

    return results


def search_web_for_travel(query, max_results=3):
    """
    Hàm tổng hợp tìm kiếm trên mạng:
    Ưu tiên DDGS -> Dự phòng Wikipedia tiếng Việt
    """
    results = search_with_ddgs(query, max_results=max_results)
    if not results:
        results = search_with_wikipedia(query)
    return results


def format_web_response(query, results):
    """
    Định dạng kết quả tìm kiếm trên mạng thành câu trả lời tự nhiên, lịch sự và dễ đọc.
    """
    if not results:
        return (
            f"🔍 Tôi đã thử tìm kiếm trên mạng về '{query}' nhưng chưa tìm thấy thông tin phù hợp. "
            f"Bạn có thể thử hỏi chi tiết hơn hoặc hỏi về các tour du lịch hiện có nhé!"
        )

    response_lines = [
        f"🌐 Tôi đã tìm kiếm trên Internet về \"{query}\" và có các thông tin sau:\n"
    ]

    for idx, r in enumerate(results[:3], start=1):
        title = r["title"]
        snippet = r["snippet"].strip()
        url = r.get("url", "")

        response_lines.append(f"📌 {idx}. {title}")
        response_lines.append(f"{snippet}")
        if url:
            response_lines.append(f"🔗 Xem thêm: {url}")
        response_lines.append("")

    response_lines.append("💡 Nếu bạn muốn tham khảo các tour du lịch trọn gói trong hệ thống (Đà Nẵng, Nha Trang, Hạ Long), hãy hỏi tôi nhé!")

    return "\n".join(response_lines).strip()
