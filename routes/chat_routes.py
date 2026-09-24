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
    - Đảm bảo ngữ cảnh multi-turn cho cả khách vãng lai (qua UUID session)
    """
    import uuid
    data = request.get_json() or {}
    question = (data.get("question") or data.get("message") or "").strip()
    session_id = data.get("session_id")
    force_web_search = bool(data.get("web_search", False))

    if not question:
        return jsonify({"answer": "Bạn hãy nhập câu hỏi để tôi có thể tư vấn nhé.", "response": "Bạn hãy nhập câu hỏi để tôi có thể tư vấn nhé."}), 400

    user_id = session.get("user_id")
    active_session_id = session_id

    # 1. Quản lý phiên hội thoại:
    # Nếu chưa có session_id:
    # - Nếu người dùng đã đăng nhập: tạo phiên mới trong MySQL DB
    # - Nếu là khách vãng lai: tạo UUID session tạm thời để duy trì multi-turn context trong RAM
    if not active_session_id:
        if user_id:
            try:
                short_title = question if len(question) <= 32 else question[:29] + "..."
                active_session_id = create_chat_session(user_id, title=short_title)
            except Exception as e:
                print("Lỗi tạo chat_session:", e)
        if not active_session_id:
            active_session_id = f"guest_{uuid.uuid4().hex[:12]}"

    # 2. AI sinh câu trả lời kết hợp DB nội bộ, Memory đa lượt và Tìm kiếm Internet
    answer = chatbot.generate_response(
        question,
        session_id=active_session_id,
        force_web_search=force_web_search
    )

    # 3. Nếu người dùng đã đăng nhập và session hợp lệ trong DB, lưu tin nhắn vào chat_history
    if user_id:
        db_sid = None
        if isinstance(active_session_id, int):
            db_sid = active_session_id
        elif isinstance(active_session_id, str) and active_session_id.isdigit():
            db_sid = int(active_session_id)

        if db_sid:
            try:
                save_chat_message(
                    db_sid,
                    question,
                    answer,
                    intent=chatbot.last_predicted_intent,
                    confidence=chatbot.last_confidence
                )
            except Exception as e:
                print("Lỗi lưu lịch sử chat vào DB:", e)

    return jsonify({
        "answer": answer,
        "response": answer,
        "session_id": active_session_id,
        "intent": chatbot.last_predicted_intent,
        "confidence": chatbot.last_confidence
    })


@chat_bp.route("/api/chat/sessions", methods=["GET"])
def api_get_sessions():
    """Lấy danh sách các phiên trò chuyện của người dùng hiện tại qua JSON."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"logged_in": False, "sessions": []})

    sessions = get_user_sessions(user_id)
    formatted = []
    for s in sessions:
        created_str = s["created_at"].strftime('%d/%m/%Y %H:%M') if s.get("created_at") else ""
        formatted.append({
            "id": s["id"],
            "title": s["title"],
            "created_at": created_str
        })

    return jsonify({"logged_in": True, "sessions": formatted})


@chat_bp.route("/api/chat/sessions/<int:session_id>/messages", methods=["GET"])
def api_get_session_messages(session_id):
    """Lấy nội dung tin nhắn của một phiên trò chuyện cụ thể qua JSON."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Vui lòng đăng nhập"}), 401

    target = get_session_by_id(session_id)
    if not target or target["user_id"] != user_id:
        return jsonify({"error": "Phiên trò chuyện không tồn tại"}), 404

    messages = get_chat_history_by_session(session_id)
    formatted = []
    for m in messages:
        t_str = m["created_at"].strftime('%H:%M') if m.get("created_at") else ""
        formatted.append({
            "id": m["id"],
            "question": m["question"],
            "answer": m["answer"],
            "time": t_str
        })

    return jsonify({
        "session_id": session_id,
        "title": target["title"],
        "messages": formatted
    })


@chat_bp.route("/api/chat/sessions/<int:session_id>", methods=["DELETE"])
def api_delete_session(session_id):
    """Xóa một phiên trò chuyện qua AJAX."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Vui lòng đăng nhập"}), 401

    target = get_session_by_id(session_id)
    if not target or target["user_id"] != user_id:
        return jsonify({"error": "Phiên không hợp lệ"}), 404

    delete_session(session_id)
    return jsonify({"success": True, "message": "Đã xóa phiên trò chuyện"})


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