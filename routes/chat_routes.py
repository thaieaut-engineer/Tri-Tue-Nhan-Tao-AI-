from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash
from ai.chatbot import Chatbot
from models.chat_session import (
    create_chat_session,
    get_user_sessions,
    get_session_by_id,
    delete_session
)
from models.chat_history import (
    save_chat_message,
    get_chat_history_by_session
)
from routes.auth_routes import login_required

chat_bp = Blueprint("chat", __name__)

# Khởi tạo thể hiện chatbot toàn cục
chatbot = Chatbot()


@chat_bp.route("/chatbot")
def chatbot_page():
    """
    Trang giao diện trò chuyện với chatbot AI.
    Hỗ trợ tiếp tục phiên chat cũ nếu có session_id và tải danh sách các phiên trước đó.
    """
    user_id = session.get("user_id")
    current_session_id = request.args.get("session_id", type=int)
    sessions = []
    messages = []

    if user_id:
        sessions = get_user_sessions(user_id)
        if current_session_id:
            target_session = get_session_by_id(current_session_id)
            if target_session and target_session["user_id"] == user_id:
                messages = get_chat_history_by_session(current_session_id)
            else:
                current_session_id = None

    return render_template(
        "chatbot.html",
        sessions=sessions,
        current_session_id=current_session_id,
        messages=messages
    )


@chat_bp.route("/api/chat", methods=["POST"])
@chat_bp.route("/chat", methods=["POST"])
def chat():
    """
    API xử lý câu hỏi từ người dùng:
    - Tìm kiếm cơ sở dữ liệu nội bộ
    - Tra cứu thời tiết thời gian thực (OpenWeatherMap / Open-Meteo)
    - Tự động tìm kiếm trên mạng Internet nếu câu hỏi không có trong DB hoặc người dùng yêu cầu
    - Lưu lịch sử hội thoại nếu đã đăng nhập
    """
    data = request.get_json() or {}
    question = (data.get("question") or data.get("message") or "").strip()
    session_id = data.get("session_id")
    force_web_search = bool(data.get("web_search", False))

    if not question:
        return jsonify({"answer": "Bạn hãy nhập câu hỏi để tôi có thể tư vấn nhé.", "response": "Bạn hãy nhập câu hỏi để tôi có thể tư vấn nhé."}), 400

    # AI sinh câu trả lời kết hợp DB nội bộ và Tìm kiếm trên Internet
    answer = chatbot.generate_response(question, force_web_search=force_web_search)

    user_id = session.get("user_id")
    active_session_id = session_id

    # Nếu người dùng đã đăng nhập, tự động lưu lịch sử
    if user_id:
        try:
            if not active_session_id:
                short_title = question if len(question) <= 30 else question[:27] + "..."
                active_session_id = create_chat_session(user_id, title=short_title)

            if active_session_id:
                save_chat_message(active_session_id, question, answer)
        except Exception as e:
            print("Lỗi lưu lịch sử chat vào DB:", e)

    return jsonify({
        "answer": answer,
        "response": answer,
        "session_id": active_session_id
    })


@chat_bp.route("/history")
@login_required
def history():
    """
    Trang xem lịch sử các phiên trò chuyện của người dùng hiện tại.
    """
    user_id = session.get("user_id")
    sessions = get_user_sessions(user_id)
    selected_id = request.args.get("session_id", type=int)

    selected_session = None
    messages = []

    if selected_id:
        target = get_session_by_id(selected_id)
        if target and target["user_id"] == user_id:
            selected_session = target
            messages = get_chat_history_by_session(selected_id)
    elif sessions and len(sessions) > 0:
        selected_session = sessions[0]
        messages = get_chat_history_by_session(selected_session["id"])

    return render_template(
        "history.html",
        sessions=sessions,
        selected_session=selected_session,
        messages=messages
    )


@chat_bp.route("/history/delete/<int:session_id>", methods=["POST"])
@login_required
def delete_chat_session_route(session_id):
    """
    Xóa một phiên trò chuyện của người dùng.
    """
    user_id = session.get("user_id")
    target = get_session_by_id(session_id)
    if target and target["user_id"] == user_id:
        delete_session(session_id)
        flash("Đã xóa phiên trò chuyện thành công.", "success")
    else:
        flash("Không thể xóa phiên trò chuyện không hợp lệ.", "danger")

    return redirect(url_for("chat.history"))