"""
ai/weather_service.py
Dịch vụ tra cứu thời tiết thời gian thực cho bất kỳ địa điểm nào.
Hỗ trợ:
 1. OpenWeatherMap API (khi cấu hình OPENWEATHER_API_KEY trong .env)
 2. Open-Meteo Global API (dự phòng tự động miễn phí, không cần API key, hỗ trợ toàn cầu)
"""

import os
import re
import requests
from dotenv import load_dotenv
from ai.preprocess import remove_accents

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()

# Bảng dịch mã thời tiết WMO (dùng cho Open-Meteo)
WMO_WEATHER_CODES = {
    0: ("Trời quang đãng, nắng đẹp", "☀️"),
    1: ("Trời hầu như không mây, có nắng nhẹ", "🌤️"),
    2: ("Mây rải rác", "⛅"),
    3: ("Trời nhiều mây, âm u", "☁️"),
    45: ("Có sương mù", "🌫️"),
    48: ("Có sương mù đọng", "🌫️"),
    51: ("Mưa phùn nhẹ", "🌦️"),
    53: ("Mưa phùn vừa", "🌦️"),
    55: ("Mưa phùn hạt dày", "🌧️"),
    61: ("Mưa rào nhẹ", "🌧️"),
    63: ("Mưa rào vừa", "🌧️"),
    65: ("Mưa to rải rác", "🌧️"),
    71: ("Có tuyết rơi nhẹ", "🌨️"),
    73: ("Có tuyết rơi vừa", "🌨️"),
    75: ("Tuyết rơi dày", "❄️"),
    80: ("Mưa rào từng cơn", "🌦️"),
    81: ("Mưa rào nặng hạt", "🌧️"),
    82: ("Mưa rào rất lớn", "⛈️"),
    95: ("Có dông, sấm sét", "⛈️"),
    96: ("Dông bão kèm mưa đá nhẹ", "⛈️"),
    99: ("Dông bão kèm mưa đá mạnh", "⛈️")
}

# Danh mục các địa danh phổ biến tại Việt Nam
KNOWN_VIETNAM_CITIES = {
    "hồ chí minh": ("Thành phố Hồ Chí Minh", "Ho Chi Minh"),
    "thành phố hồ chí minh": ("Thành phố Hồ Chí Minh", "Ho Chi Minh"),
    "tp hcm": ("Thành phố Hồ Chí Minh", "Ho Chi Minh"),
    "tphcm": ("Thành phố Hồ Chí Minh", "Ho Chi Minh"),
    "sài gòn": ("Thành phố Hồ Chí Minh", "Ho Chi Minh"),
    "hà nội": ("Hà Nội", "Hanoi"),
    "đà nẵng": ("Đà Nẵng", "Da Nang"),
    "nha trang": ("Nha Trang", "Nha Trang"),
    "hạ long": ("Hạ Long", "Ha Long"),
    "hải phòng": ("Hải Phòng", "Hai Phong"),
    "cần thơ": ("Cần Thơ", "Can Tho"),
    "đà lạt": ("Đà Lạt", "Da Lat"),
    "sa pa": ("Sa Pa", "Sa Pa"),
    "sapa": ("Sa Pa", "Sa Pa"),
    "phú quốc": ("Phú Quốc", "Phu Quoc"),
    "quy nhơn": ("Quy Nhơn", "Quy Nhon"),
    "huế": ("Huế", "Hue"),
    "hội an": ("Hội An", "Hoi An"),
    "vũng tàu": ("Vũng Tàu", "Vung Tau"),
    "côn đảo": ("Côn Đảo", "Con Dao"),
    "phan thiết": ("Phan Thiết", "Phan Thiet"),
    "mũi né": ("Mũi Né", "Mui Ne"),
    "hà giang": ("Hà Giang", "Ha Giang"),
    "mộc châu": ("Mộc Châu", "Moc Chau"),
    "ninh bình": ("Ninh Bình", "Ninh Binh"),
    "tam đảo": ("Tam Đảo", "Tam Dao"),
    "quảng ninh": ("Quảng Ninh", "Quang Ninh"),
    "quảng bình": ("Quảng Bình", "Quang Binh"),
    "bến tre": ("Bến Tre", "Ben Tre"),
    "an giang": ("An Giang", "An Giang"),
    "cà mau": ("Cà Mau", "Ca Mau")
}


def extract_city_from_question(question):
    """
    Trích xuất tên hiển thị và tên tìm kiếm quốc tế của địa danh từ câu hỏi.
    Trả về: (display_name, query_name)
    """
    if not question:
        return None, None
    q_low = question.lower().strip()

    # 1. So khớp từ điển thành phố phổ biến (ưu tiên từ khóa dài trước)
    sorted_cities = sorted(KNOWN_VIETNAM_CITIES.items(), key=lambda x: len(x[0]), reverse=True)
    for key, (display_name, query_name) in sorted_cities:
        if key in q_low:
            return display_name, query_name

    # 2. Nhận diện theo các mẫu câu hỏi tự nhiên
    patterns = [
        # Mẫu 1: thời tiết / nhiệt độ ở [địa danh] thế nào
        r"(?:thời tiết|nhiệt độ|dự báo thời tiết)(?:\s+(?:ở|tại|khu vực|thành phố|tỉnh))?\s+([A-Za-zÀ-ỹ0-9\s]+?)(?:\s+(?:thế nào|như thế nào|hôm nay|ngày mai|có mưa không|ra sao|hiện tại|bao nhiêu)|\?|$)",
        # Mẫu 2: ở / tại [địa danh] có mưa / có lạnh không
        r"(?:ở|tại)\s+([A-Za-zÀ-ỹ0-9\s]+?)\s+(?:có mưa|có lạnh|có nóng|mưa không|lạnh không|nóng không|thời tiết|nhiệt độ|hôm nay)",
        # Mẫu 3: [địa danh] có mưa không / có lạnh không
        r"^([A-Za-zÀ-ỹ0-9\s]+?)\s+(?:có mưa không|có mưa ko|mưa không|mưa ko|có lạnh không|lạnh không|có nóng không|nóng không|bao nhiêu độ|thời tiết thế nào)"
    ]

    for pat in patterns:
        match = re.search(pat, q_low)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"^(thành phố|tỉnh|khu vực|ở|tại)\s+", "", candidate).strip()
            if candidate and len(candidate) >= 2 and candidate not in ["hôm nay", "ngày mai", "hiện tại", "nào", "đâu"]:
                display = candidate.title()
                query = remove_accents(candidate)
                return display, query

    return None, None


def fetch_weather_openweathermap(city_name, query_name, api_key):
    """
    Tra cứu thời tiết qua OpenWeatherMap API
    """
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": query_name or city_name,
            "appid": api_key,
            "units": "metric",
            "lang": "vi"
        }
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            main = data.get("main", {})
            weather = data.get("weather", [{}])[0]
            wind = data.get("wind", {})
            desc_text = weather.get("description", "").capitalize()
            weather_main = weather.get("main", "").lower()
            is_rain = "rain" in weather_main or "drizzle" in weather_main or "thunderstorm" in weather_main or "mưa" in desc_text.lower()

            return {
                "source": "OpenWeatherMap",
                "city": city_name,
                "temp": round(main.get("temp", 0), 1),
                "feels_like": round(main.get("feels_like", 0), 1),
                "humidity": main.get("humidity", 0),
                "description": desc_text,
                "raw_desc": desc_text,
                "wind_speed": round(wind.get("speed", 0) * 3.6, 1),
                "is_raining": is_rain
            }
    except Exception as e:
        print("Lỗi OpenWeatherMap API:", e)

    return None


def fetch_weather_openmeteo(city_name, query_name):
    """
    Dự phòng tra cứu thời tiết qua Open-Meteo Global API (Miễn phí, không cần key)
    """
    try:
        # Bước 1: Geocoding
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        search_term = query_name if query_name else remove_accents(city_name)
        geo_params = {
            "name": search_term,
            "count": 1,
            "format": "json"
        }
        geo_resp = requests.get(geo_url, params=geo_params, timeout=5).json()
        results = geo_resp.get("results", [])

        if not results:
            return None

        top_geo = results[0]
        lat = top_geo["latitude"]
        lon = top_geo["longitude"]

        # Bước 2: Thời tiết
        weather_url = "https://api.open-meteo.com/v1/forecast"
        w_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "timezone": "auto"
        }
        w_resp = requests.get(weather_url, params=w_params, timeout=5).json()
        current = w_resp.get("current", {})

        w_code = current.get("weather_code", 0)
        desc, emoji = WMO_WEATHER_CODES.get(w_code, ("Thời tiết bình thường", "🌤️"))
        precip = current.get("precipitation", 0)
        is_rain = (w_code in [51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99]) or (precip > 0) or ("mưa" in desc.lower())

        return {
            "source": "Open-Meteo",
            "city": city_name,
            "temp": round(current.get("temperature_2m", 0), 1),
            "feels_like": round(current.get("apparent_temperature", 0), 1),
            "humidity": current.get("relative_humidity_2m", 0),
            "description": f"{emoji} {desc}",
            "raw_desc": desc,
            "wind_speed": round(current.get("wind_speed_10m", 0), 1),
            "precipitation": precip,
            "is_raining": is_rain
        }
    except Exception as e:
        print("Lỗi Open-Meteo API:", e)

    return None


def get_api_key():
    """
    Đọc API key từ file weatherapi.py ở thư mục gốc.
    Nếu file weatherapi.py trống, không tồn tại hoặc lỗi, fallback sang biến môi trường .env hoặc rỗng.
    """
    key = ""
    # 1. Thử import từ module weatherapi
    try:
        import weatherapi
        key = getattr(weatherapi, "OPENWEATHER_API_KEY", getattr(weatherapi, "API_KEY", ""))
    except Exception:
        key = ""

    # 2. Nếu file weatherapi.py trống hoặc chỉ chứa chuỗi key thuần túy
    if not key:
        file_path = os.path.join(os.path.dirname(__file__), "..", "weatherapi.py")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content and "=" not in content and not content.startswith("#"):
                        key = content
            except Exception:
                pass

    # 3. Fallback sang file .env
    if not key:
        key = os.getenv("OPENWEATHER_API_KEY", "").strip()

    return key if isinstance(key, str) else ""


def get_weather_for_location(city_name, query_name=None):
    """
    Hàm tổng hợp tra cứu thời tiết:
    - Đọc API key từ file weatherapi.py (nơi người dùng có thể cấu hình API key).
    - Nếu có key: gọi OpenWeatherMap API.
    - Nếu file weatherapi.py trống hoặc gọi OpenWeatherMap thất bại:
      -> Tự động fallback dự phòng sang Open-Meteo Global API (an toàn, không lộ key khi commit GitHub).
    """
    api_key = get_api_key()
    weather_data = None

    if api_key:
        weather_data = fetch_weather_openweathermap(city_name, query_name, api_key)

    # Nếu file trống, không có key hoặc gọi API thất bại -> Tự động chuyển sang fallback
    if not weather_data:
        weather_data = fetch_weather_openmeteo(city_name, query_name)

    return weather_data


def format_weather_response(city_name, w_data, question=None):
    """
    Sinh câu trả lời thời tiết thông minh dựa trên câu hỏi cụ thể của người dùng.
    Không dùng văn mẫu cố định mà trực tiếp giải đáp thắc mắc (mưa/nắng, độ lạnh, đi chơi, trang phục...).
    """
    if not w_data:
        return (
            f"🌤️ Hiện tại tôi chưa thể lấy được dữ liệu thời tiết cho khu vực '{city_name}'. "
            f"Bạn có thể thử kiểm tra lại tên địa danh hoặc thử lại sau ít phút nhé!"
        )

    temp = w_data["temp"]
    feels_like = w_data["feels_like"]
    humidity = w_data["humidity"]
    raw_desc = w_data.get("raw_desc", "")
    desc = w_data.get("description", "")
    wind = w_data["wind_speed"]
    city = w_data["city"]
    is_raining = w_data.get("is_raining", False)

    # Chuẩn hóa mô tả bầu trời thành văn phong tự nhiên
    if raw_desc:
        clean_desc = raw_desc.strip().lower()
    else:
        clean_desc = re.sub(r"^[^\w\s]+", "", desc).strip().lower()

    # Loại bỏ tiền tố "trời" hoặc "bầu trời" nếu đã có sẵn để tránh bị lặp từ
    clean_desc = re.sub(r"^(trời|bầu trời)\s+", "", clean_desc).strip()

    q_low = (question or "").lower()

    # 1. Câu hỏi về mưa / tạnh / mang ô dù
    is_rain_inquiry = any(k in q_low for k in [
        "có mưa không", "có mưa ko", "mưa không", "mưa ko", "trời mưa",
        "mưa to", "mưa rào", "mưa dông", "mưa hay nắng", "nắng hay mưa",
        "tạnh", "mang ô", "mang dù", "áo mưa"
    ])
    if is_rain_inquiry:
        if is_raining:
            return (
                f"Hiện tại ở **{city} đang có {clean_desc}** bạn nhé! "
                f"Nhiệt độ ngoài trời khoảng **{temp}°C**, độ ẩm khá cao ({humidity}%) và gió {wind} km/h. "
                f"Nếu bạn chuẩn bị ra ngoài thì nhớ mang theo ô (dù) hoặc áo mưa để tránh bị ướt nhé."
            )
        else:
            return (
                f"Hiện tại ở **{city} không có mưa** bạn nhé! "
                f"Bầu trời lúc này **{clean_desc}**, nhiệt độ ghi nhận khoảng **{temp}°C** (cảm giác thực tế như {feels_like}°C), "
                f"độ ẩm {humidity}% và gió nhẹ {wind} km/h. Thời tiết rất thuận lợi để bạn ra ngoài dạo phố hoặc tham quan."
            )

    # 2. Câu hỏi về độ lạnh / rét
    is_cold_inquiry = any(k in q_low for k in [
        "có lạnh không", "lạnh không", "lạnh ko", "có rét không",
        "trời lạnh", "rét không", "lạnh lắm không", "lạnh buốt"
    ])
    if is_cold_inquiry:
        if temp <= 18:
            return (
                f"Ở **{city} hiện tại khá lạnh** bạn nhé! "
                f"Nhiệt độ đo được là **{temp}°C** (cảm giác ngoài trời như **{feels_like}°C**), trời {clean_desc} và độ ẩm {humidity}%. "
                f"Nếu ra ngoài bạn nên chuẩn bị thêm áo khoác ấm hoặc khăn choàng để giữ ấm cơ thể, đặc biệt vào sáng sớm hoặc chiều tối."
            )
        elif 19 <= temp <= 24:
            return (
                f"Thời tiết ở **{city} hiện tại se lạnh mát mẻ chứ không quá buốt** đâu bạn. "
                f"Nhiệt độ đang ở mức **{temp}°C** (cảm nhận như {feels_like}°C), trời {clean_desc} với gió {wind} km/h. "
                f"Bạn chỉ cần mang theo một chiếc áo khoác mỏng hoặc cardigan là đã rất thoải mái rồi."
            )
        else:
            return (
                f"Ở **{city} hiện tại không lạnh** đâu bạn nhé. "
                f"Thời tiết khá ấm áp với nhiệt độ khoảng **{temp}°C** (cảm giác thực tế {feels_like}°C), trời {clean_desc}. "
                f"Bạn có thể mặc trang phục thường ngày thoải mái mà không cần áo ấm."
            )

    # 3. Câu hỏi về nắng nóng / oi bức
    is_hot_inquiry = any(k in q_low for k in [
        "có nóng không", "nóng không", "nóng ko", "oi bức",
        "trời nóng", "nóng bức", "nắng gắt", "nóng lắm không"
    ])
    if is_hot_inquiry:
        if temp >= 32:
            return (
                f"Hiện tại ở **{city} khá nắng nóng** bạn nhé. "
                f"Nhiệt độ ngoài trời đo được là **{temp}°C**, cảm giác thực tế oi bức khoảng **{feels_like}°C** với độ ẩm {humidity}%. "
                f"Bạn nên hạn chế ở ngoài trời quá lâu vào buổi trưa, nhớ bôi kem chống nắng và uống nhiều nước nhé."
            )
        else:
            return (
                f"Thời tiết ở **{city} lúc này không bị quá nóng** đâu bạn. "
                f"Nhiệt độ duy trì quanh mức **{temp}°C** (cảm giác ngoài trời {feels_like}°C), trời {clean_desc}, "
                f"độ ẩm {humidity}% và gió nhẹ {wind} km/h, tương đối dễ chịu cho việc đi lại."
            )

    # 4. Câu hỏi về nhiệt độ cụ thể
    is_temp_inquiry = any(k in q_low for k in [
        "bao nhiêu độ", "mấy độ", "đo được bao nhiêu độ", "bao nhiu độ"
    ]) or ("nhiệt độ" in q_low and any(w in q_low for w in ["bao nhiêu", "mấy", "thế nào", "bao nhiu", "hiện tại", "ra sao"]))
    if is_temp_inquiry:
        return (
            f"Nhiệt độ hiện tại tại **{city} là {temp}°C** "
            f"(cảm giác thực tế ngoài trời khoảng **{feels_like}°C**). "
            f"Tình trạng bầu trời lúc này {clean_desc}, độ ẩm không khí {humidity}% và tốc độ gió khoảng {wind} km/h."
        )

    # 5. Câu hỏi có nên đi chơi / thời tiết có đẹp không
    is_outing_inquiry = any(k in q_low for k in [
        "đi chơi được không", "đi chơi đc ko", "có nên đi chơi",
        "đi dạo được không", "thời tiết có đẹp không", "thời tiết đẹp không",
        "thích hợp đi chơi", "ra ngoài được không", "tham quan được không"
    ])
    if is_outing_inquiry:
        if is_raining or wind >= 25:
            return (
                f"Thời điểm này ở **{city} thời tiết chưa thật sự lý tưởng để đi chơi ngoài trời** "
                f"vì hiện đang {clean_desc}, độ ẩm {humidity}% và gió {wind} km/h. "
                f"Nếu muốn ra ngoài, bạn nên chọn các địa điểm trong nhà như bảo tàng, quán cà phê hoặc trung tâm mua sắm và nhớ mang theo ô dù nhé!"
            )
        else:
            return (
                f"Thời tiết ở **{city} hôm nay rất lý tưởng để đi chơi và tham quan ngoại cảnh**! "
                f"Bầu trời lúc này **{clean_desc}**, không có mưa, nhiệt độ duy trì ở mức dễ chịu khoảng **{temp}°C** "
                f"(cảm nhận {feels_like}°C) và gió nhẹ {wind} km/h. Rất thuận tiện để bạn dạo phố và chụp ảnh kỷ niệm."
            )

    # 6. Câu hỏi về trang phục / mặc gì
    is_outfit_inquiry = any(k in q_low for k in [
        "mặc gì", "nên mặc gì", "mặc đồ gì", "chuẩn bị đồ gì", "trang phục"
    ])
    if is_outfit_inquiry:
        if temp <= 18:
            return (
                f"Với thời tiết se lạnh khoảng **{temp}°C** tại **{city}** lúc này, bạn nên chọn **trang phục giữ ấm** "
                f"như áo len, áo khoác dày, quần dài và khăn quàng, đặc biệt cần thiết khi ra ngoài vào sáng sớm hoặc tối muộn."
            )
        elif 19 <= temp <= 25:
            return (
                f"Thời tiết ở **{city}** hiện đang mát mẻ quanh mức **{temp}°C**. Bạn nên mặc **trang phục thoải mái "
                f"kèm một chiếc áo khoác mỏng hoặc cardigan nhẹ nhàng**, vừa tiện di chuyển vừa đủ ấm khi có gió nhẹ."
            )
        else:
            return (
                f"Nhiệt độ tại **{city}** đang ở mức **{temp}°C** khá ấm áp. Bạn nên chọn **trang phục thoáng mát, "
                f"thấm hút mồ hôi tốt** (như áo phông, quần short, váy nhẹ) và nhớ mang thêm mũ nón, kính râm để che nắng khi di chuyển nhé."
            )

    # 7. Câu hỏi thời tiết tổng quan (Không dùng văn mẫu cố định)
    if is_raining:
        return (
            f"🌤️ Hiện tại ở **{city}** đang có **{clean_desc}**, nhiệt độ vào khoảng **{temp}°C** và độ ẩm không khí khá cao ({humidity}%). "
            f"Gió nhẹ khoảng {wind} km/h. Nếu bạn có kế hoạch ra ngoài tham quan hôm nay, hãy chuẩn bị sẵn ô (dù) hoặc áo mưa để chuyến đi không bị gián đoạn nhé!"
        )
    elif temp <= 18:
        return (
            f"🌤️ Thời tiết tại **{city}** lúc này se lạnh đặc trưng, nhiệt độ khoảng **{temp}°C** (cảm giác ngoài trời như **{feels_like}°C**), "
            f"bầu trời **{clean_desc}**. Độ ẩm không khí ở mức {humidity}% và gió thổi nhẹ {wind} km/h. "
            f"Không khí rất trong lành và dễ chịu, bạn chỉ cần mang thêm một chiếc áo khoác ấm là có thể thoải mái dạo chơi ngắm cảnh rồi!"
        )
    elif temp >= 32:
        return (
            f"🌤️ Ở **{city}** thời điểm này thời tiết khá nắng ấm và oi nhẹ, nhiệt độ khoảng **{temp}°C** (cảm giác thực tế ngoài trời khoảng **{feels_like}°C**), "
            f"trời **{clean_desc}**. Độ ẩm {humidity}% và gió {wind} km/h. Bạn nhớ mang theo kem chống nắng, kính râm và bổ sung đủ nước khi tham gia các hoạt động ngoài trời nhé."
        )
    else:
        return (
            f"🌤️ Thời tiết ở **{city}** hiện tại khá lý tưởng, bầu trời **{clean_desc}** và không có mưa. "
            f"Nhiệt độ duy trì quanh mức **{temp}°C** (cảm giác thực tế dễ chịu như **{feels_like}°C**), "
            f"độ ẩm {humidity}% cùng làn gió nhẹ {wind} km/h. Rất thuận lợi cho các hoạt động tham quan và vui chơi ngoài trời trong ngày hôm nay!"
        )


def is_weather_query(question):
    """Kiểm tra câu hỏi có liên quan đến thời tiết, nhiệt độ, mưa nắng, đi lại hay không"""
    if not question:
        return False
    q_low = question.lower()

    # Các từ khóa thời tiết trực tiếp
    weather_keywords = [
        "thời tiết", "dự báo thời tiết", "nhiệt độ", "bao nhiêu độ", "mấy độ",
        "có mưa không", "có mưa ko", "mưa không", "mưa ko", "trời mưa", "mưa to",
        "mưa rào", "mưa dông", "mưa hay nắng", "nắng hay mưa", "trời có mưa",
        "có lạnh không", "lạnh không", "lạnh ko", "trời lạnh", "có rét không", "rét không",
        "có nóng không", "nóng không", "nóng ko", "trời nóng", "oi bức",
        "có nắng không", "nắng không", "trời nắng", "nắng gắt",
        "thời tiết đẹp không", "thời tiết có đẹp", "thời tiết đi chơi", "thích hợp đi chơi",
        "đi chơi được không", "thời tiết dạo này"
    ]
    if any(kw in q_low for kw in weather_keywords):
        return True

    # Câu hỏi trang phục kết hợp địa danh/thời tiết
    if ("mặc gì" in q_low or "mặc đồ gì" in q_low) and any(w in q_low for w in ["thời tiết", "trời", "lạnh", "nóng", "nắng", "mưa", "hôm nay"]):
        return True

    return False
