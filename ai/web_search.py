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


try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


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


def fetch_page_main_content(url, timeout=5):
    """
    TRUY CẬP TRỰC TIẾP TRANG WEB VÀ BÓC TÁCH NỘI DUNG CHÍNH (DEEP WEB CONTENT EXTRACTION):
    - Sử dụng requests và BeautifulSoup để tải nội dung HTML thực tế từ các nguồn uy tín.
    - Loại bỏ các thành phần rác (quảng cáo, script, navigation, header, footer).
    - Trích xuất các đoạn văn bản có ý nghĩa thực tế để giải đáp trực tiếp cho du khách.
    """
    if not BS4_AVAILABLE or not url or not url.startswith("http"):
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi,en;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200 or not resp.content:
            return []

        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "figure", "noscript", "iframe"]):
            tag.decompose()

        paragraphs = []
        for p in soup.find_all("p"):
            txt = clean_html(p.get_text(strip=True))
            if len(txt) >= 40:
                is_junk = any(j in txt.lower() for j in [
                    "chính sách bảo mật", "điều khoản sử dụng", "đăng ký nhận tin",
                    "tất cả quyền được bảo lưu", "copyright", "chia sẻ bài viết",
                    "theo dõi chúng tôi", "bấm vào đây để", "xem chi tiết tại",
                    "quảng cáo", "liên hệ quảng cáo"
                ])
                if not is_junk and txt not in paragraphs:
                    paragraphs.append(txt)
                    if len(paragraphs) >= 20:
                        break
        return paragraphs
    except Exception:
        return []


def extract_relevant_deep_facts(paragraphs, query, max_facts=3):
    """
    LỌC CÁC ĐOẠN VĂN SÁT TRỌNG TÂM CÂU HỎI NHẤT TỪ NỘI DUNG TRANG WEB:
    - Chấm điểm từng đoạn văn bản dựa trên mật độ từ khóa và tính hướng dẫn/giải pháp.
    - Ưu tiên các đoạn hướng dẫn hành động (quy trình, bước làm, lưu ý).
    - Loại bỏ các câu chuyện cá nhân trong phần bình luận hoặc hỏi đáp độc giả.
    """
    if not paragraphs:
        return []

    q_words = set(re.findall(r'\w+', query.lower()))
    scored_paragraphs = []

    advisory_clues = [
        "bước", "cách", "xử lý", "báo", "liên hệ", "khách sạn", "cảnh sát", "công an",
        "hồ sơ", "giấy tờ", "bảo hiểm", "bình tĩnh", "kiểm tra", "trình báo", "hướng dẫn",
        "kinh nghiệm", "lưu ý", "khuyên", "quy định", "giải quyết", "thủ tục", "lịch trình"
    ]

    personal_junk = [
        "tôi có liên hệ", "tôi đã gọi", "mời đối tượng", "tục tỉu", "kỷ càng",
        "nữ diễn viên", "ông park", "kiều oanh", "xin hỏi luật sư"
    ]

    for p in paragraphs:
        p_low = p.lower()
        if any(j in p_low for j in personal_junk):
            continue

        score = sum(1.5 for w in q_words if len(w) > 2 and w in p_low)
        score += sum(1.2 for clue in advisory_clues if clue in p_low)

        # Ưu tiên các câu mang tính chỉ dẫn, có đánh số thứ tự hoặc gạch đầu dòng
        if re.match(r'^\s*(?:bước\s*\d+|thứ\s*(?:nhất|hai|ba)|\d+[\.\)]|[•\-*])', p_low):
            score += 2.5
        elif any(lead in p_low for lead in ["cần làm", "hãy", "nên", "trước tiên", "quy trình", "ngay lập tức"]):
            score += 1.5

        if 50 <= len(p) <= 350:
            score += 1.0
        scored_paragraphs.append((score, p))

    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)

    results = []
    for sc, p in scored_paragraphs:
        if sc > 0:
            results.append(p)
            if len(results) >= max_facts:
                break

    if not results and paragraphs:
        for p in paragraphs:
            if not any(j in p.lower() for j in personal_junk):
                results.append(p)
                if len(results) >= max_facts:
                    break

    return results


def reformulate_travel_query(query):
    """
    ĐỊNH HƯỚNG TỪ KHÓA TÌM KIẾM DU LỊCH THÔNG MINH (AI TRAVEL QUERY REFORMULATION):
    - Nhận diện tình huống sự cố, an toàn, pháp lý du lịch để tránh tìm sai lệch sang y khoa tâm thần.
    - Khử nhập nhằng địa danh (ví dụ 'Hồ Chí Minh' -> 'TP Hồ Chí Minh Sài Gòn').
    - Bổ sung ngữ cảnh du lịch để máy tìm kiếm trả về danh thắng, khách sạn, ẩm thực, an toàn.
    """
    q = query.strip()
    q_low = q.lower()

    # 1. Nhận diện tình huống sự cố, rủi ro và an toàn du lịch
    theft_keywords = ["ăn cắp", "an cap", "trộm cắp", "trom cap", "mất cắp", "mat cap", "móc túi", "moc tui", "bị cướp", "bi cuop", "mất đồ", "mat do", "mất ví", "mat vi", "mất tài sản", "mất vali", "thất lạc hành lý"]
    if any(k in q_low for k in theft_keywords):
        if any(h in q_low for h in ["khách sạn", "ks", "phòng"]):
            return "kinh nghiệm quy trình xử lý khi bị mất cắp mất đồ ở khách sạn khi đi du lịch"
        return "kinh nghiệm hướng dẫn các bước xử lý khi bị mất cắp tài sản khi đi du lịch"

    if any(k in q_low for k in ["mất hộ chiếu", "mat ho chieu", "mất cccd", "mat cccd", "mất chứng minh", "mất giấy tờ"]):
        return "thủ tục xử lý khi bị mất hộ chiếu căn cước công dân khi đi du lịch"

    if any(k in q_low for k in ["ngộ độc", "ngo doc", "đau bụng", "dị ứng hải sản", "say xe", "say sóng"]):
        return "cách xử lý sơ cứu khi bị ngộ độc thực phẩm say sóng khi đi du lịch"

    if any(k in q_low for k in ["chặt chém", "chat chem", "ép giá", "ep gia", "lừa đảo", "lua dao", "đường dây nóng du lịch"]):
        return "cách xử lý và số điện thoại phản ánh chặt chém lừa đảo du khách"

    if any(k in q_low for k in ["hướng dẫn viên", "hdv"]) and any(k in q_low for k in ["tiếng", "ngôn ngữ", "ngoại ngữ", "đoàn"]):
        return "tiêu chuẩn hướng dẫn viên du lịch ngoại ngữ tiếng anh tiếng pháp đoàn khách quốc tế"

    # 2. Khử nhập nhằng địa danh Hồ Chí Minh
    if any(k in q_low for k in ["hồ chí minh", "ho chi minh", "hcm", "tphcm"]):
        q = re.sub(r"\bhồ chí minh\b", "TP Hồ Chí Minh", q, flags=re.IGNORECASE)
        q = re.sub(r"\bho chi minh\b", "TP Hồ Chí Minh", q, flags=re.IGNORECASE)
        q = re.sub(r"\btphcm\b", "TP Hồ Chí Minh", q, flags=re.IGNORECASE)
        q = re.sub(r"\bhcm\b", "TP Hồ Chí Minh", q, flags=re.IGNORECASE)
        if "sài gòn" not in q_low and "sai gon" not in q_low:
            q += " Sài Gòn"

    # 3. Bổ sung từ khóa du lịch theo ý định
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


def search_web_for_travel(query, max_results=3, deep_fetch=True):
    """
    Hàm tổng hợp tìm kiếm và bóc tách nội dung chuyên sâu trên mạng (Deep Web Content Extraction):
    1. Định hướng từ khóa du lịch & an toàn (Query Reformulation).
    2. Tìm kiếm DuckDuckGo / Wikipedia tiếng Việt.
    3. Truy cập trực tiếp trang web đích bằng BeautifulSoup để bóc tách nội dung thật,
       không dừng lại ở việc chỉ lấy liên kết đơn thuần.
    """
    reformulated = reformulate_travel_query(query)
    results = search_with_ddgs(reformulated, max_results=max_results)

    # Nếu truy vấn định hướng chưa có kết quả, thử truy vấn nguyên bản
    if not results and query != reformulated:
        results = search_with_ddgs(query, max_results=max_results)

    # Dự phòng Wikipedia nếu DuckDuckGo bị chặn kết nối
    if not results:
        results = search_with_wikipedia(reformulated)

    # BÓC TÁCH NỘI DUNG SÂU (DEEP SCRAPING) TỪ TRANG WEB THẬT
    if deep_fetch and results and BS4_AVAILABLE:
        for r in results[:2]:
            url = r.get("url", "")
            if url and url.startswith("http") and not url.endswith((".pdf", ".doc", ".docx")):
                deep_paragraphs = fetch_page_main_content(url, timeout=5)
                if deep_paragraphs:
                    r["deep_content"] = extract_relevant_deep_facts(deep_paragraphs, query, max_facts=3)

    return results


# ====================================================================
# CƠ SỞ TRI THỨC DU LỊCH CHUYÊN SÂU CÁC ĐIỂM NGOÀI HỆ THỐNG
# ====================================================================
CURATED_DESTINATIONS_KNOWLEDGE = {
    "thái lan": {
        "title": "Cẩm Nang Du Lịch Thái Lan (Xứ Sở Chùa Vàng)",
        "highlights": [
            "Thủ đô Bangkok: Chiêm bái Chùa Vàng Wat Phra Kaew, Cung điện Hoàng Gia Grand Palace, Chùa Wat Arun bên sông Chao Phraya, mua sắm thả ga tại chợ đêm Jodd Fairs và IconSiam.",
            "Thành phố biển Pattaya: Vui chơi tại Đảo san hô Koh Larn nước trong vắt, tham quan Chợ nổi 4 miền và show diễn nghệ thuật Alcazar rực rỡ.",
            "Chiang Mai & Chiang Rai: Không gian cổ kính yên bình, check-in Chùa Trắng Wat Rong Khun độc nhất vô nhị và trải nghiệm chăm sóc voi thân thiện.",
            "Thiên đường biển Phuket: Khám phá Vịnh Maya, Đảo Phi Phi lừng danh thế giới với các hoạt động lặn biển ngắm san hô."
        ],
        "food": "Tom Yum cay nồng đậm vị, Pad Thai tôm tươi, Xôi xoài cốt dừa béo ngậy, Nộm đu đủ Som Tum giòn cay, Trà sữa Thái đỏ và thiên đường hải sản tươi sống tại chợ đêm.",
        "best_time": "Từ tháng 11 đến tháng 4 năm sau (thời tiết khô ráo, nắng ấm mát mẻ). Đặc biệt vào giữa tháng 4 có Lễ hội té nước truyền thống Songkran vô cùng náo nhiệt.",
        "budget_tips": "Tour trọn gói 5N4Đ dao động từ 6.500.000 - 8.500.000 VNĐ/khách. Công dân Việt Nam được miễn thị thực (visa) du lịch 30 ngày chỉ cần hộ chiếu còn hạn trên 6 tháng."
    },
    "nhật bản": {
        "title": "Cẩm Nang Du Lịch Nhật Bản (Xứ Sở Mặt Trời Mọc)",
        "highlights": [
            "Thủ đô Tokyo hiện đại: Ngã tư Shibuya nhộn nhịp nhất hành tinh, Đền cổ Senso-ji Asakusa ngàn năm tuổi, tháp truyền hình Tokyo Skytree, phố điện tử Akihabara.",
            "Biểu tượng Núi Phú Sĩ: Ngắm đỉnh núi tuyết phủ hùng vĩ, dạo bước trong Làng cổ Oshino Hakkai dưới chân núi với nguồn nước khoáng tinh khiết.",
            "Cố đô Kyoto trầm mặc: Chiêm bái Chùa Vàng Kinkaku-ji lấp lánh, ngắm Rừng trúc Arashiyama xanh mướt, check-in hàng ngàn cổng Torii đỏ thắm tại Đền Fushimi Inari.",
            "Thành phố Osaka: Khám phá Lâu đài Osaka uy nghiêm, thiên đường ẩm thực đường phố Dotonbori rực rỡ bảng đèn neon."
        ],
        "food": "Bò Wagyu/Kobe nướng tan chảy trong miệng, Sushi & Sashimi hải sản tươi rói từ chợ cá, Mì Ramen nước dùng hầm xương đậm đà, Bánh bạch tuộc Takoyaki và rượu Sake truyền thống.",
        "best_time": "Mùa xuân (cuối tháng 3 đến giữa tháng 4) ngắm hoa anh đào Sakura nở rộ khắp các công viên; hoặc Mùa thu (tháng 10 đến tháng 11) chiêm ngưỡng mùa lá đỏ Momiji lãng mạn.",
        "budget_tips": "Tour trọn gói 5N4Đ - 6N5Đ dao động từ 24.000.000 - 32.000.000 VNĐ/khách. Khách du lịch Việt Nam cần nộp hồ sơ xin visa du lịch trước chuyến đi từ 3 - 4 tuần."
    },
    "hàn quốc": {
        "title": "Cẩm Nang Du Lịch Hàn Quốc (Xứ Sở Kim Chi)",
        "highlights": [
            "Thủ đô Seoul: Mặc trang phục Hanbok truyền thống check-in Cung điện Gyeongbokgung, ngắm toàn cảnh thành phố từ Tháp Namsan, dạo Phố đi bộ Myeongdong và Làng cổ Bukchon Hanok.",
            "Đảo Nami mộng mơ: Địa danh ghi dấu bộ phim 'Bản tình ca mùa đông' với những hàng cây ngân hạnh rực rỡ sắc vàng sắc đỏ.",
            "Đảo ngọc Jeju: Kỳ quan thiên nhiên thế giới với Đỉnh núi lửa Seongsan Ilchulbong ngắm bình minh và Thác nước Cheonjiyeon nguyên sơ.",
            "Thành phố cảng Busan: Check-in Làng bích họa Gamcheon đầy màu sắc, chợ cá Jagalchi lớn nhất xứ Hàn và bãi biển Haeundae sôi động."
        ],
        "food": "Thịt nướng Samgyeopsal cuốn lá kim và kim chi cay nồng, Gà rán sốt cay kèm bia tươi Chimaek, Cơm trộn thố đá Bibimbap, Canh sườn bò hầm sâm, Bánh gạo cay Tokbokki.",
        "best_time": "Mùa thu (tháng 9 đến tháng 11) - mùa đẹp nhất trong năm với tiết trời se lạnh mát mẻ và rừng lá phong rực đỏ; hoặc Mùa xuân (tháng 3 đến tháng 5) ngắm hoa anh đào.",
        "budget_tips": "Chi phí tour trọn gói 5N4Đ dao động từ 12.000.000 - 16.500.000 VNĐ/khách (cần làm thủ tục xin visa du lịch Hàn Quốc)."
    },
    "singapore": {
        "title": "Cẩm Nang Du Lịch Singapore & Malaysia",
        "highlights": [
            "Singapore hiện đại & xanh sạch: Quần thể siêu cây khổng lồ Gardens by the Bay, Vịnh Marina Bay Sands, Đảo giải trí Sentosa & Universal Studios, tượng Sư tử biển Merlion.",
            "Thủ đô Kuala Lumpur (Malaysia): Chụp ảnh cùng Tháp đôi Petronas Twin Towers biểu tượng, viếng Động đá vôi Batu huyền bí, khu phức hợp giải trí Casino Cao nguyên Genting."
        ],
        "food": "Cua sốt ớt Chilli Crab đậm đà ăn kèm bánh bao chiên, Cơm gà Hải Nam chuẩn vị, Cháo ếch Geylang, Trà sữa kéo bọt Teh Tarik, Mì Laksa hải sản cay béo.",
        "best_time": "Khí hậu nhiệt đới quanh năm thuận tiện du lịch; thời điểm lý tưởng nhất là từ tháng 11 đến tháng 2 năm sau (thời tiết mát hơn và rực rỡ không khí lễ hội). Miễn visa cho khách Việt.",
        "budget_tips": "Tour liên tuyến Singapore - Malaysia 5N4Đ trọn gói khoảng 9.500.000 - 12.500.000 VNĐ/khách."
    },
    "trung quốc": {
        "title": "Cẩm Nang Du Lịch Trung Quốc (Đại Lục Cảnh Sắc Kỳ Vĩ)",
        "highlights": [
            "Bắc Kinh - Thượng Hải: Chinh phục Vạn Lý Trường Thành kỳ quan thế giới, thăm Tử Cấm Thành uy nghiêm, ngắm cảnh Bến Thượng Hải hoa lệ bên dòng sông Hoàng Phố.",
            "Phượng Hoàng Cổ Trấn: Thị trấn cổ hơn 1300 năm tuổi với dãy nhà sàn Điếu Cước Lâu soi bóng huyền ảo xuống dòng sông Đà Giang.",
            "Trương Gia Giới: Công viên rừng quốc gia với hàng ngàn cột đá sa thạch dựng đứng (bối cảnh phim Avatar), chinh phục Cổng Trời Thiên Môn Sơn và Cầu kính 7D đáy vực."
        ],
        "food": "Vịt quay Bắc Kinh da giòn óng ả, Lẩu Tứ Xuyên cay tê kích thích vị giác, Bánh bao Tiểu Long Bao tràn nước súp ngọt thanh, Đậu phụ Ma Bà Tứ Xuyên.",
        "best_time": "Tháng 4 - tháng 5 (mùa xuân nắng ấm) và tháng 9 - tháng 11 (mùa thu vàng lá đỏ rực rỡ, không khí mát dịu dễ chịu).",
        "budget_tips": "Tour trọn gói dao động từ 11.500.000 - 18.500.000 VNĐ/khách (cần làm visa đoàn hoặc visa cá nhân)."
    },
    "hà nội": {
        "title": "Cẩm Nang Du Lịch Thủ Đô Hà Nội (Nét Đẹp Ngàn Năm Văn Hiến)",
        "highlights": [
            "Quanh Hồ Hoàn Kiếm: Dạo phố đi bộ cuối tuần, viếng Đền Ngọc Sơn cổ kính, Cầu Thê Húc cong cong như con tôm màu son.",
            "36 Phố Phường: Lạc bước trong những ngõ nhỏ rêu phong, phố nghề truyền thống và ngắm kiến trúc Pháp cổ điển.",
            "Di tích lịch sử thiêng liêng: Lăng Chủ tịch Hồ Chí Minh, Chùa Một Cột, Văn Miếu - Quốc Tử Giám trường đại học đầu tiên, Hoàng Thành Thăng Long di sản thế giới.",
            "Hồ Tây thơ mộng: Viếng Chùa Trấn Quốc cổ nhất Hà Nội, ngắm hoàng hôn lãng mạn tại bến Phủ Tây Hồ và ngắm Cầu Long Biên chứng nhân lịch sử."
        ],
        "food": "Phở bò gia truyền (Bát Đàn, Thìn Lò Đúc), Bún chả Hàng Mành nướng than hoa, Chả cá Lã Vọng thơm lừng ngải cứu thì là, Bún thang, Bánh cuốn Thanh Trì, Cà phê trứng Giảng béo ngậy.",
        "best_time": "Mùa thu Hà Nội (từ tháng 9 đến tháng 11) - thời điểm lãng mạn và đẹp nhất trong năm với tiết trời mát mẻ, nắng vàng hanh hao và thoang thoảng hương hoa sữa nồng nàn.",
        "budget_tips": "Chi phí du lịch tự túc hoặc tour ngắn ngày 2N1Đ - 3N2Đ khoảng 2.000.000 - 3.500.000 VNĐ/người (chưa gồm vé máy bay)."
    },
    "hồ chí minh": {
        "title": "Cẩm Nang Du Lịch TP Hồ Chí Minh (Sài Gòn Trẻ Trung & Năng Động)",
        "highlights": [
            "Kiến trúc biểu tượng: Nhà thờ Đức Bà, Bưu điện Trung tâm thành phố, Dinh Độc Lập lịch sử, Chợ Bến Thành sầm uất trăm năm tuổi.",
            "Hiện đại & Sống động: Tòa tháp Landmark 81 cao nhất Việt Nam ngắm trọn vẹn thành phố từ tầng cao, phố đi bộ Nguyễn Huệ, dạo du thuyền ngắm hoàng hôn sông Sài Gòn.",
            "Cuộc sống về đêm (Nightlife): Hòa mình vào không khí sôi động của Phố Tây Bùi Viện, thưởng thức ẩm thực đường phố và ngắm cảnh đêm rực rỡ ánh đèn."
        ],
        "food": "Cơm tấm sườn bì chả nướng mỡ hành, Hủ tiếu Nam Vang đậm đà, Bánh mì Huỳnh Hoa ngập tràn nhân thịt nguội, Ốc đêm vỉa hè Quận 4, Bánh tráng trộn, Cà phê sữa đá vợt truyền thống.",
        "best_time": "Từ tháng 12 đến tháng 4 năm sau (mùa khô trời nắng ráo, không khí dễ chịu, rất thích hợp du xuân và tham quan ngoài trời).",
        "budget_tips": "Chuyến đi 2 - 3 ngày tại TP.HCM dao động khoảng 2.500.000 - 4.000.000 VNĐ/người."
    },
    "quảng bình": {
        "title": "Cẩm Nang Du Lịch Quảng Bình (Vương Quốc Hang Động Kỳ Vĩ)",
        "highlights": [
            "Kỳ quan Vườn Quốc gia Phong Nha - Kẻ Bàng di sản thế giới: Động Thiên Đường tráng lệ với nhũ đá triệu năm tuổi, Động Phong Nha đi thuyền trên sông ngầm ngắm thạch nhũ.",
            "Trải nghiệm mạo hiểm & sông nước: Sông Chày - Hang Tối (đu dây zipline vượt sông, chèo thuyền kayak và tắm bùn tự nhiên phục hồi sức khỏe), Suối Nước Moọc nước xanh ngọc bích.",
            "Danh thắng biển cát: Đồi cát trắng Quang Phú trượt cát cực đã, viếng Vũng Chùa - Đảo Yến nơi an nghỉ của Đại tướng Võ Nguyên Giáp, tắm biển Nhật Lệ hoang sơ."
        ],
        "food": "Bánh bột lọc mệ Xuân dẻo dai đậm vị tôm thịt, Cháo canh cá lóc nóng hổi ăn kèm ram giòn, Đẻn biển nướng sả ớt, Khoai deo Ba Đồn thơm bùi, Mực nhảy biển Nhật Lệ tươi ngọt.",
        "best_time": "Từ tháng 4 đến tháng 8 là mùa lý tưởng nhất: trời nắng ráo, nước sông trong vắt mát lạnh, các hoạt động khám phá hang động và bơi lội diễn ra thuận lợi.",
        "budget_tips": "Tour khám phá Quảng Bình 3N2Đ có chi phí khoảng 2.800.000 - 3.800.000 VNĐ/người."
    },
    "hải phòng": {
        "title": "Cẩm Nang Du Lịch Hải Phòng (Hoa Phượng Đỏ & Thiên Đường Food Tour)",
        "highlights": [
            "Biển đảo kỳ vĩ: Quần đảo Cát Bà, Vịnh Lan Hạ hoang sơ tuyệt mỹ (chèo thuyền kayak Hang Sáng Hang Tối, tắm biển Đảo Khỉ), bãi biển Đồ Sơn.",
            "Danh thắng văn hóa: Tuyệt Tình Cốc Núi Đèo nước xanh ngọc bích, Tháp Tường Long, Ga Hải Phòng kiến trúc Pháp cổ kính, Nhà hát Lớn thành phố."
        ],
        "food": "Thiên đường Food Tour đình đám: Bánh đa cua bể đậm đà gạch son, Bánh mì que cay giòn rụm chấm tương ớt Chí Chương, Chè dừa dầm béo ngậy, Bì bò, Sủi dìn nóng hổi, Ốc xào dừa chợ Cố Đạo.",
        "best_time": "Tháng 4 đến tháng 10 (mùa hoa phượng nở đỏ thắm khắp phố phường, thời tiết biển Cát Bà trong lành lý tưởng để nghỉ dưỡng).",
        "budget_tips": "Chuyến Food Tour 2N1Đ chi phí siêu tiết kiệm chỉ khoảng 1.200.000 - 1.800.000 VNĐ/người."
    },
    "tam đảo": {
        "title": "Cẩm Nang Du Lịch Tam Đảo (Thị Trấn Bồng Bềnh Trong Sương)",
        "highlights": [
            "Săn mây & Check-in: Cầu Mây Tam Đảo săn biển mây bồng bềnh, Quán Gió ngắm toàn cảnh thung lũng mờ sương, Nhà thờ Đá cổ kính phong cách Gothic cổ điển, Thác Bạc nguyên sơ.",
            "Chinh phục đỉnh cao: Tháp truyền hình Tam Đảo cao hơn 1400 bậc đá, Cổng Trời mở ra toàn cảnh núi rừng hùng vĩ."
        ],
        "food": "Ngọn su su non xào tỏi giòn ngọt trứ danh, Gà đồi nướng đất sét thơm phức, Lợn mán nướng than hồng xiên que, Trứng gà nướng, Rượu sâu chít đậm đà.",
        "best_time": "Quanh năm (chỉ cách Hà Nội hơn 70km), khí hậu mát mẻ 4 mùa trong 1 ngày, cực kỳ thích hợp cho chuyến đi nghỉ dưỡng xả hơi cuối tuần 2 ngày 1 đêm.",
        "budget_tips": "Chi phí chuyến đi 2N1Đ cuối tuần tự túc khoảng 1.200.000 - 1.800.000 VNĐ/người."
    },
    "cà mau": {
        "title": "Cẩm Nang Du Lịch Cà Mau (Vùng Đất Mũi Cực Nam Thiêng Liêng)",
        "highlights": [
            "Cột mốc Cực Nam Tổ Quốc: Check-in Mốc tọa độ quốc gia GPS 0001 tại Đất Mũi, Biểu tượng con tàu no gió vươn ra biển Đông, Cột cờ Hà Nội tại Mũi Cà Mau.",
            "Sinh thái rừng ngập mặn: Vườn quốc gia Mũi Cà Mau, Rừng tràm U Minh Hạ bạt ngàn với trải nghiệm đi vỏ lãi len lỏi dưới tán đước, Khu du lịch Hòn Đá Bạc."
        ],
        "food": "Cua biển Cà Mau chắc nịch nhiều gạch nức tiếng cả nước, Cá thòi lòi nướng muối ớt, Vọp nướng mỡ hành, Bồn bồn muối chua xào tôm nõn, Tôm tít rang muối.",
        "best_time": "Từ tháng 12 đến tháng 4 năm sau (mùa khô sông nước êm đềm, cua biển vào mùa ngọt thịt nhất).",
        "budget_tips": "Chuyến đi 2N1Đ - 3N2Đ từ TP.HCM khoảng 2.200.000 - 3.200.000 VNĐ/người."
    },
    "đài loan": {
        "title": "Cẩm Nang Du Lịch Đài Loan (Xứ Sở Trà Sữa & Chợ Đêm)",
        "highlights": [
            "Thủ đô Đài Bắc: Tòa tháp Taipei 101 biểu tượng, Viện Bảo tàng Cố Cung lưu giữ bảo vật ngàn năm, Quảng trường Tưởng Giới Thạch.",
            "Làng cổ & Thiên nhiên: Thả đèn trời cầu bình an tại Làng cổ Thập Phần (Shifen), Làng cổ Cửu Phần (Jiufen) bối cảnh hoạt hình Vùng Đất Linh Hồn, Hồ Nhật Nguyệt thơ mộng."
        ],
        "food": "Trà sữa trân chú nguồn gốc Đài Loan, Tiểu long bao Din Tai Fung ngập súp ngọt, Đậu phụ thối chiên giòn, Bánh mì kẹp thịt Quan Tài, Mì bò hầm Đài Bắc.",
        "best_time": "Tháng 9 đến tháng 11 (mùa thu mát mẻ, nắng nhẹ) và tháng 2 đến tháng 4 (mùa hoa anh đào nở rộ). Khách Việt có thể xin E-visa tiên tiến dễ dàng.",
        "budget_tips": "Tour trọn gói 5N4Đ dao động từ 10.500.000 - 14.500.000 VNĐ/khách."
    },
    "bali": {
        "title": "Cẩm Nang Du Lịch Bali (Thiên Đường Nghỉ Dưỡng Nhiệt Đới Indonesia)",
        "highlights": [
            "Đền đài linh thiêng: Đền Uluwatu tọa lạc trên vách đá cheo leo ngắm hoàng hôn biển rực lửa, Cổng trời Lempuyang hùng vĩ, Đền Nước Nông Pura Ulun Danu bên hồ Bratan.",
            "Check-in & Thiên nhiên: Trải nghiệm xích đu Bali Swing giữa thung lũng nhiệt đới, Ruộng bậc thang Tegalalang xanh mướt, Bãi biển Kuta và Nusa Dua."
        ],
        "food": "Vịt chiên giòn Bebek Goreng, Cơm sườn heo nướng thảo mộc Naughty Nuri's, Cơm truyền thống Nasi Campur, Heo quay Babi Guling giòn da.",
        "best_time": "Từ tháng 5 đến tháng 10 (mùa khô trời trong xanh, ít mưa, sóng biển êm đềm rất thích hợp lướt sóng và nghỉ dưỡng). Miễn visa cho khách du lịch Việt Nam.",
        "budget_tips": "Tour trọn gói 4N3Đ - 5N4Đ khoảng 9.500.000 - 13.500.000 VNĐ/khách."
    },
    "châu âu": {
        "title": "Cẩm Nang Du Lịch Châu Âu (Hành Trình Di Sản Cổ Kính)",
        "highlights": [
            "Pháp: Tháp Eiffel, Bảo tàng Louvre, Khải Hoàn Môn, Du thuyền trên sông Seine thơ mộng.",
            "Ý & Vatican: Đấu trường La Mã Colosseum, Đài phun nước Trevi, Tòa thánh Vatican tráng lệ, Thành phố kênh đào Venice.",
            "Thụy Sĩ: Đỉnh núi tuyết Titlis / Jungfraujoch nóc nhà Châu Âu, Hồ Geneva trong vắt.",
            "Đức: Lâu đài cổ tích Neuschwanstein, Cổng thành Brandenburg tại thủ đô Berlin."
        ],
        "food": "Bò bít tết Pháp sốt tiêu đen kèm rượu vang, Pizza Napoletana nướng củi và Mì Ý Pasta truyền thống, Kem Gelato Ý, Xúc xích nướng Đức kèm bia tươi.",
        "best_time": "Từ tháng 5 đến tháng 10: thời tiết nắng ấm rực rỡ, ngày dài hơn đêm, phong cảnh thiên nhiên rực rỡ sắc màu.",
        "budget_tips": "Tour liên tuyến Châu Âu (Pháp - Thụy Sĩ - Ý) 9 - 11 ngày dao động từ 55.000.000 - 75.000.000 VNĐ/khách (cần nộp hồ sơ xin visa Schengen trước 1.5 - 2 tháng)."
    },
    "phú thọ": {
        "title": "Cẩm Nang Du Lịch Phú Thọ (Về Miền Đất Tổ Thiêng Liêng)",
        "highlights": [
            "Khu Di tích Lịch sử Đền Hùng: Hành hương lên núi Nghĩa Lĩnh viếng Đền Hạ, Đền Trung, Đền Thượng và Lăng Hùng Vương tưởng nhớ công lao các vua Hùng dựng nước.",
            "Thiên nhiên xanh mát: Đồi chè Long Cốc (được mệnh danh là 'ốc đảo chè' đẹp nhất Việt Nam với hàng trăm đồi bát úp nhấp nhô trong mây), Vườn quốc gia Xuân Sơn hoang sơ kỳ bí."
        ],
        "food": "Thịt chua Thanh Sơn giòn sần sật cuộn lá sung chấm tương ớt, Cá thính chua Bá Xuyên, Bánh tai Phú Thọ, Búp chè non chiên giòn, Rau sắn nấu cá chua.",
        "best_time": "Tháng 3 âm lịch (dịp Giỗ Tổ Hùng Vương 10/3 AL) để hòa mình vào đại lễ văn hóa; hoặc từ tháng 9 đến tháng 12 để săn mây tuyệt đẹp tại Đồi chè Long Cốc.",
        "budget_tips": "Chuyến đi 1 - 2 ngày tự túc hoặc tour ngắn từ Hà Nội khoảng 900.000 - 1.500.000 VNĐ/người."
    },
    "lý sơn": {
        "title": "Cẩm Nang Du Lịch Đảo Lý Sơn (Vương Quốc Tỏi & Trầm Tích Núi Lửa)",
        "highlights": [
            "Di sản địa chất triệu năm: Cổng Tò Vò bằng nham thạch tự nhiên độc nhất vô nhị, Đỉnh Thới Lới ngắm trọn vẹn biển đảo từ miệng núi lửa đã tắt, Hang Câu vách đá hùng vĩ.",
            "Đảo Bé (An Bình): Bãi tắm nước trong vắt nhìn thấu tận đáy san hô, trải nghiệm lặn biển và chèo thuyền thúng."
        ],
        "food": "Gỏi rong biển Lý Sơn, Gỏi tỏi tươi thơm nồng bùi ngậy, Cua huỳnh đế hấp, Ốc xà cừ xào sả ớt, Chả cá Lý Sơn chiên nóng hổi.",
        "best_time": "Từ tháng 4 đến tháng 8 là mùa biển êm, trời trong xanh và nắng đẹp nhất để đi tàu cao tốc ra đảo.",
        "budget_tips": "Chuyến đi 3N2Đ kết hợp khám phá Quảng Ngãi - Lý Sơn khoảng 2.800.000 - 3.800.000 VNĐ/người."
    },
    "mai châu": {
        "title": "Cẩm Nang Du Lịch Mai Châu - Hòa Bình (Thung Lũng Yên Bình)",
        "highlights": [
            "Bản Lác & Bản Pom Coọng: Đạp xe giữa những cánh đồng lúa xanh mướt ngát hương, trải nghiệm ở nhà sàn người Thái trắng và múa sạp giao lưu văn nghệ về đêm.",
            "Cảnh quan kỳ vĩ: Đèo Thung Khe (Đèo Đá Trắng) quanh năm mây mù che phủ như trời Âu, Hang Mỏ Luông kỳ thú."
        ],
        "food": "Cơm lam nướng ống tre chấm muối vừng, Gà đồi nướng mắc khén thơm nức, Thịt lợn mán nướng xiên than hồng, Xôi nếp nương ngũ sắc dẻo thơm, Rượu cần Mai Châu.",
        "best_time": "Tháng 3 - 4 (mùa hoa ban trắng nở rộ khắp thung lũng) hoặc tháng 9 - 10 (mùa lúa chín vàng óng ả thơm ngát hương lúa mới).",
        "budget_tips": "Chuyến đi 2N1Đ cuối tuần thư giãn xả stress chỉ khoảng 1.100.000 - 1.600.000 VNĐ/người."
    },
    "bến tre": {
        "title": "Cẩm Nang Du Lịch Bến Tre (Xứ Sở Dừa Xanh Mát)",
        "highlights": [
            "Du lịch sinh thái cồn: Cồn Phụng, Cồn Quy (đi xuồng ba lá len lỏi dưới rặng dừa nước rợp bóng mát, nghe đờn ca tài tử Nam Bộ và thưởng thức trà mật ong hoa nhãn).",
            "Làng nghề truyền thống: Ghé thăm các lò kẹo dừa thủ công nóng hổi thơm lừng, xưởng đan lát mỹ nghệ từ thân dừa."
        ],
        "food": "Cơm dừa nấu nước dừa xiêm ăn kèm tôm rang cốt dừa, Cá tai tượng chiên xù cuốn bánh tráng rau sống, Gỏi củ hủ dừa tôm thịt giòn ngọt thanh tao, Bánh xèo hến ốc gạo Cồn Phú Đa.",
        "best_time": "Từ tháng 5 đến tháng 8 là mùa các vườn cây ăn trái (chôm chôm, sầu riêng, măng cụt) vào vụ trĩu quả ngon ngọt nhất.",
        "budget_tips": "Chuyến đi 1 - 2 ngày từ Sài Gòn khoảng 800.000 - 1.800.000 VNĐ/người."
    }
}

# Ánh xạ alias cho các địa danh phổ biến
DESTINATION_ALIASES = {
    "sài gòn": "hồ chí minh",
    "tphcm": "hồ chí minh",
    "hcm": "hồ chí minh",
    "tp hồ chí minh": "hồ chí minh",
    "tp hcm": "hồ chí minh",
    "phong nha": "quảng bình",
    "kẻ bàng": "quảng bình",
    "malaysia": "singapore",
    "taiwan": "đài loan",
    "đền hùng": "phú thọ",
    "long cốc": "phú thọ",
    "bản lác": "mai châu",
    "hòa bình": "mai châu",
    "cồn phụng": "bến tre",
    "pháp": "châu âu",
    "ý": "châu âu",
    "đức": "châu âu",
    "anh": "châu âu"
}


def get_curated_destination_guide(dest_name):
    """Lấy hướng dẫn du lịch chi tiết có sẵn theo tên địa danh."""
    if not dest_name:
        return None
    d_clean = dest_name.lower().strip()
    target_key = DESTINATION_ALIASES.get(d_clean, d_clean)
    return CURATED_DESTINATIONS_KNOWLEDGE.get(target_key)


def synthesize_travel_search_response(query, results, destination_name=None):
    """
    TỔNG HỢP VÀ ĐIỀU CHẾ TRI THỨC TÌM KIẾM THÔNG MINH (AI TRAVEL SYNTHESIS ENGINE):
    Thay vì chỉ dán danh sách đường link thô sơ, hàm này:
    1. Nhận diện địa danh đối sánh với cơ sở tri thức du lịch chuyên sâu.
    2. Chắt lọc các điểm nhấn thắng cảnh, ẩm thực, thời điểm vàng và chi phí.
    3. Kết hợp trích lọc thông tin tươi mới từ kết quả tìm kiếm web.
    4. Trả về bài tư vấn hoàn chỉnh, giàu trải nghiệm, chuyên nghiệp.
    5. Đặt các đường dẫn tham khảo gọn gàng ở cuối bài.
    """
    dest_key = destination_name.lower().strip() if destination_name else None
    guide = get_curated_destination_guide(dest_key)

    if not guide and dest_key:
        for k in CURATED_DESTINATIONS_KNOWLEDGE.keys():
            if k in dest_key or dest_key in k:
                guide = CURATED_DESTINATIONS_KNOWLEDGE[k]
                break

    lines = []

    # TRƯỜNG HỢP 1: Có tri thức chuyên sâu đã được kiểm định
    if guide:
        lines.append(f"🌟 **{guide['title']}**\n")
        lines.append("📍 **Các điểm tham quan & trải nghiệm biểu tượng:**")
        for hl in guide["highlights"]:
            lines.append(f"  • {hl}")
        lines.append("")

        lines.append(f"🍜 **Ẩm thực & đặc sản trứ danh:**\n  {guide['food']}\n")
        lines.append(f"🌤️ **Thời điểm lý tưởng nhất:**\n  {guide['best_time']}\n")
        lines.append(f"💰 **Chi phí dự kiến & Thủ tục:**\n  {guide['budget_tips']}\n")

        # Bổ sung các nguồn web tham khảo nếu có
        if results:
            lines.append("📚 **Nguồn thông tin & cẩm nang tham khảo thêm:**")
            for r in results[:2]:
                title = r.get("title", "").strip()
                url = r.get("url", "").strip()
                if title and url:
                    lines.append(f"  • [{title}]({url})")
            lines.append("")

        lines.append("💡 *Lời khuyên tư vấn: Để có chuyến du lịch an toàn và trọn vẹn, bạn hãy lưu ý theo dõi sát điều kiện thời tiết, bảo quản kỹ tư trang và tìm hiểu trước các quy định văn hóa của điểm đến nhé!*")
        return "\n".join(lines)

    # TRƯỜNG HỢP 2: Tổng hợp từ kết quả bóc tách trang web & máy tìm kiếm
    if results:
        deep_facts = []
        extracted_facts = []
        source_links = []

        for r in results:
            title = r.get("title", "").strip()
            snippet = clean_html(r.get("snippet", "")).strip()
            url = r.get("url", "").strip()

            if url and title:
                source_links.append(f"[{title}]({url})")

            # Thu thập các sự kiện chuyên sâu được cào bóc trực tiếp từ trang web
            for df in r.get("deep_content", []):
                if df not in deep_facts:
                    deep_facts.append(df)

            # Phân tách thành từng câu hoàn chỉnh từ snippet
            sentences = re.split(r'(?<=[.!?])\s+', snippet)
            for s in sentences:
                s_clean = s.strip()
                is_rhetorical_question = s_clean.endswith('?')
                is_intro_junk = any(junk in s_clean.lower() for junk in [
                    "xem thêm", "xem chi tiết", "xem full", "bấm vào đây", "đăng ký ngay",
                    "hãy cùng", "bài viết này", "sẽ giúp bạn", "dưới đây là", "bạn đang phân vân",
                    "cùng tìm hiểu", "bạn đã biết", "bạn có biết", "nên đi đâu để", "vậy tháng", "vậy mùa"
                ])
                if len(s_clean) > 25 and not is_rhetorical_question and not is_intro_junk:
                    extracted_facts.append(s_clean)

        lines.append(f"🌐 **Tư Vấn Thông Tin Du Lịch: \"{query}\"**\n")

        # Ưu tiên hiển thị nội dung chuyên sâu được trích xuất trực tiếp từ trang web
        facts_to_display = deep_facts[:4] if deep_facts else extracted_facts[:4]
        if facts_to_display:
            lines.append("📋 **Nội dung hướng dẫn & giải đáp trọng tâm:**")
            for fact in facts_to_display:
                lines.append(f"• {fact}")
            lines.append("")

        if source_links:
            lines.append("📚 **Nguồn trang web đã tra cứu & trích xuất:**")
            for sl in source_links[:3]:
                lines.append(f"• {sl}")
            lines.append("")

        lines.append("💡 *Lời khuyên tư vấn: Trong mọi tình huống sự cố hoặc du lịch thực tế, bạn hãy luôn giữ bình tĩnh, bảo quản cẩn thận giấy tờ tùy thân và liên hệ ngay với người phụ trách hoặc cơ quan chức năng để được hỗ trợ kịp thời nhé!*")
        return "\n".join(lines)

    # TRƯỜNG HỢP 3: Không có kết quả
    return (
        f"🔍 Rất tiếc, tôi chưa tìm thấy thông tin phù hợp cho thắc mắc '{query}'. "
        f"Bạn có thể miêu tả chi tiết hơn câu hỏi hoặc hỏi về kinh nghiệm chuẩn bị hành lý, an toàn du lịch, thời tiết và các điểm đến nhé!"
    )


def format_web_response(query, results, destination_name=None):
    """
    Định dạng kết quả tìm kiếm theo phong cách trợ lý tư vấn du lịch thông minh,
    tổng hợp kiến thức sâu sắc thay vì chỉ trích xuất đường link đơn thuần.
    """
    return synthesize_travel_search_response(query, results, destination_name=destination_name)

