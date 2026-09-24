from flask import Blueprint, render_template, request, redirect, url_for, flash
from routes.auth_routes import admin_required
from models.tour import (
    get_all_tours, get_tour_by_id, create_tour, update_tour, delete_tour, count_tours
)
from models.tour_schedule import (
    get_tour_schedule, add_schedule_item, delete_schedule_item
)
from models.category import (
    get_all_categories, create_category, update_category, delete_category, count_categories
)
from models.user import (
    get_all_users, update_user_role, delete_user, count_users
)
from models.qa_data import (
    get_all_qa, get_qa_by_id, create_qa, update_qa, delete_qa, count_qa, get_distinct_intents
)
from models.chat_history import get_all_chat_logs, count_chat_messages
from models.chat_session import count_sessions
from models.booking import (
    get_all_bookings, update_booking_status, delete_booking, count_bookings, get_total_revenue
)
from models.review import (
    get_all_reviews, delete_review
)
from routes.chat_routes import chatbot
from ai.self_learning import self_learning_engine

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


# =======================================================
# 1. DASHBOARD TỔNG QUAN
# =======================================================
@admin_bp.route("/")
@admin_bp.route("/dashboard")
@admin_required
def dashboard():
    stats = {
        "tours": count_tours(),
        "users": count_users(),
        "categories": count_categories(),
        "qa": count_qa(),
        "messages": count_chat_messages(),
        "sessions": count_sessions(),
        "bookings": count_bookings(),
        "pending_bookings": count_bookings("pending"),
        "revenue": get_total_revenue()
    }
    recent_tours = get_all_tours()[:5]
    recent_bookings = get_all_bookings()[:5]
    recent_logs = get_all_chat_logs(limit=5)

    return render_template(
        "admin/dashboard.html",
        stats=stats,
        recent_tours=recent_tours,
        recent_bookings=recent_bookings,
        recent_logs=recent_logs
    )


# =======================================================
# 2. QUẢN LÝ TOUR DU LỊCH
# =======================================================
@admin_bp.route("/tours")
@admin_required
def tours_management():
    tours = get_all_tours()
    categories = get_all_categories()
    return render_template("admin/tours.html", tours=tours, categories=categories)


@admin_bp.route("/tours/create", methods=["POST"])
@admin_required
def tour_create():
    name = request.form.get("name", "").strip()
    category_id = request.form.get("category_id", type=int)
    destination = request.form.get("destination", "").strip()
    duration = request.form.get("duration", "").strip()
    price = request.form.get("price", type=float)
    description = request.form.get("description", "").strip()
    image = request.form.get("image", "").strip()

    if not name or not destination or not duration or price is None:
        flash("Vui lòng nhập đầy đủ các thông tin bắt buộc của tour.", "danger")
        return redirect(url_for("admin.tours_management"))

    new_id = create_tour(name, category_id, destination, duration, price, description, image)
    if new_id:
        flash("Thêm tour mới thành công!", "success")
    else:
        flash("Thêm tour thất bại.", "danger")

    return redirect(url_for("admin.tours_management"))


@admin_bp.route("/tours/edit/<int:tour_id>", methods=["POST"])
@admin_required
def tour_edit(tour_id):
    name = request.form.get("name", "").strip()
    category_id = request.form.get("category_id", type=int)
    destination = request.form.get("destination", "").strip()
    duration = request.form.get("duration", "").strip()
    price = request.form.get("price", type=float)
    description = request.form.get("description", "").strip()
    image = request.form.get("image", "").strip()

    if not name or not destination or not duration or price is None:
        flash("Vui lòng nhập đầy đủ các thông tin bắt buộc.", "danger")
        return redirect(url_for("admin.tours_management"))

    if update_tour(tour_id, name, category_id, destination, duration, price, description, image):
        flash("Cập nhật thông tin tour thành công!", "success")
    else:
        flash("Cập nhật tour thất bại.", "danger")

    return redirect(url_for("admin.tours_management"))


@admin_bp.route("/tours/delete/<int:tour_id>", methods=["POST"])
@admin_required
def tour_delete(tour_id):
    if delete_tour(tour_id):
        flash("Đã xóa tour thành công!", "success")
    else:
        flash("Xóa tour thất bại.", "danger")
    return redirect(url_for("admin.tours_management"))


@admin_bp.route("/tours/<int:tour_id>/schedule", methods=["GET", "POST"])
@admin_required
def tour_schedule_management(tour_id):
    tour = get_tour_by_id(tour_id)
    if not tour:
        flash("Tour không tồn tại.", "danger")
        return redirect(url_for("admin.tours_management"))

    if request.method == "POST":
        day_number = request.form.get("day_number", type=int)
        location = request.form.get("location", "").strip()
        activity = request.form.get("activity", "").strip()
        description = request.form.get("description", "").strip()

        if day_number and location and activity:
            add_schedule_item(tour_id, day_number, location, activity, description)
            flash(f"Đã thêm lịch trình ngày {day_number}!", "success")
        else:
            flash("Vui lòng điền đủ thông tin ngày, địa điểm và hoạt động.", "danger")
        return redirect(url_for("admin.tour_schedule_management", tour_id=tour_id))

    schedule = get_tour_schedule(tour_id)
    return render_template("admin/tour_schedule.html", tour=tour, schedule=schedule)


@admin_bp.route("/tours/schedule/delete/<int:schedule_id>", methods=["POST"])
@admin_required
def schedule_delete(schedule_id):
    tour_id = request.form.get("tour_id", type=int)
    delete_schedule_item(schedule_id)
    flash("Đã xóa mục lịch trình thành công.", "success")
    if tour_id:
        return redirect(url_for("admin.tour_schedule_management", tour_id=tour_id))
    return redirect(url_for("admin.tours_management"))


# =======================================================
# 3. QUẢN LÝ DANH MỤC TOUR
# =======================================================
@admin_bp.route("/categories")
@admin_required
def categories_management():
    categories = get_all_categories()
    return render_template("admin/categories.html", categories=categories)


@admin_bp.route("/categories/create", methods=["POST"])
@admin_required
def category_create():
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()

    if not name:
        flash("Tên danh mục không được để trống.", "danger")
        return redirect(url_for("admin.categories_management"))

    if create_category(name, description):
        flash("Thêm danh mục mới thành công!", "success")
    else:
        flash("Thêm danh mục thất bại (có thể đã trùng tên).", "danger")

    return redirect(url_for("admin.categories_management"))


@admin_bp.route("/categories/edit/<int:category_id>", methods=["POST"])
@admin_required
def category_edit(category_id):
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()

    if not name:
        flash("Tên danh mục không được để trống.", "danger")
        return redirect(url_for("admin.categories_management"))

    if update_category(category_id, name, description):
        flash("Cập nhật danh mục thành công!", "success")
    else:
        flash("Cập nhật danh mục thất bại.", "danger")

    return redirect(url_for("admin.categories_management"))


@admin_bp.route("/categories/delete/<int:category_id>", methods=["POST"])
@admin_required
def category_delete(category_id):
    if delete_category(category_id):
        flash("Đã xóa danh mục thành công!", "success")
    else:
        flash("Xóa danh mục thất bại.", "danger")
    return redirect(url_for("admin.categories_management"))


# =======================================================
# 4. QUẢN LÝ CÂU HỎI MẪU & CÂU TRẢ LỜI (AI Q&A)
# =======================================================
@admin_bp.route("/qa")
@admin_required
def qa_management():
    qa_list = get_all_qa()
    tours = get_all_tours()
    intents = get_distinct_intents()
    return render_template(
        "admin/qa_data.html",
        qa_list=qa_list,
        tours=tours,
        intents=intents
    )


@admin_bp.route("/qa/create", methods=["POST"])
@admin_required
def qa_create():
    question = request.form.get("question", "").strip()
    answer = request.form.get("answer", "").strip()
    intent = request.form.get("intent", "").strip()
    tour_id = request.form.get("tour_id", type=int)

    if not question or not answer or not intent:
        flash("Vui lòng nhập đầy đủ câu hỏi, câu trả lời và ý định (intent).", "danger")
        return redirect(url_for("admin.qa_management"))

    if create_qa(question, answer, intent, tour_id):
        # Tự động nạp lại chatbot
        chatbot.reload()
        flash("Thêm câu hỏi mẫu thành công và đã cập nhật mô hình AI!", "success")
    else:
        flash("Thêm câu hỏi mẫu thất bại.", "danger")

    return redirect(url_for("admin.qa_management"))


@admin_bp.route("/qa/edit/<int:qa_id>", methods=["POST"])
@admin_required
def qa_edit(qa_id):
    question = request.form.get("question", "").strip()
    answer = request.form.get("answer", "").strip()
    intent = request.form.get("intent", "").strip()
    tour_id = request.form.get("tour_id", type=int)

    if not question or not answer or not intent:
        flash("Vui lòng nhập đầy đủ thông tin bắt buộc.", "danger")
        return redirect(url_for("admin.qa_management"))

    if update_qa(qa_id, question, answer, intent, tour_id):
        chatbot.reload()
        flash("Cập nhật câu hỏi đáp thành công và đã cập nhật AI!", "success")
    else:
        flash("Cập nhật thất bại.", "danger")

    return redirect(url_for("admin.qa_management"))


@admin_bp.route("/qa/delete/<int:qa_id>", methods=["POST"])
@admin_required
def qa_delete(qa_id):
    if delete_qa(qa_id):
        chatbot.reload()
        flash("Đã xóa câu hỏi đáp và cập nhật AI thành công!", "success")
    else:
        flash("Xóa thất bại.", "danger")
    return redirect(url_for("admin.qa_management"))


@admin_bp.route("/qa/retrain", methods=["POST"])
@admin_required
def qa_retrain():
    """
    Huấn luyện lại mô hình AI từ dữ liệu mới nhất trong cơ sở dữ liệu.
    """
    try:
        chatbot.reload()
        flash("✅ Mô hình AI đã được huấn luyện lại thành công với dữ liệu mới nhất!", "success")
    except Exception as e:
        flash(f"Lỗi khi huấn luyện lại AI: {e}", "danger")
    return redirect(url_for("admin.qa_management"))


# =======================================================
# 5. QUẢN LÝ TÀI KHOẢN NGƯỜI DÙNG
# =======================================================
@admin_bp.route("/users")
@admin_required
def users_management():
    users = get_all_users()
    return render_template("admin/users.html", users=users)


@admin_bp.route("/users/role/<int:user_id>", methods=["POST"])
@admin_required
def user_change_role(user_id):
    role = request.form.get("role", "").strip()
    if update_user_role(user_id, role):
        flash("Cập nhật vai trò người dùng thành công!", "success")
    else:
        flash("Cập nhật vai trò thất bại.", "danger")
    return redirect(url_for("admin.users_management"))


@admin_bp.route("/users/delete/<int:user_id>", methods=["POST"])
@admin_required
def user_delete_route(user_id):
    if delete_user(user_id):
        flash("Đã xóa tài khoản người dùng thành công!", "success")
    else:
        flash("Xóa người dùng thất bại.", "danger")
    return redirect(url_for("admin.users_management"))


# =======================================================
# 6. XEM LỊCH SỬ HỎI ĐÁP TOÀN HỆ THỐNG
# =======================================================
@admin_bp.route("/history")
@admin_required
def history_logs():
    logs = get_all_chat_logs(limit=200)
    return render_template("admin/history.html", logs=logs)


# =======================================================
# 7. QUẢN LÝ ĐƠN ĐẶT TOUR (BOOKINGS)
# =======================================================
@admin_bp.route("/bookings")
@admin_required
def bookings_management():
    status = request.args.get("status", "").strip() or None
    search = request.args.get("search", "").strip() or None

    bookings_list = get_all_bookings(status=status, search=search)
    stats = {
        "total": count_bookings(),
        "pending": count_bookings("pending"),
        "confirmed": count_bookings("confirmed"),
        "completed": count_bookings("completed"),
        "cancelled": count_bookings("cancelled"),
        "revenue": get_total_revenue()
    }

    return render_template(
        "admin/bookings.html",
        bookings=bookings_list,
        stats=stats,
        current_status=status,
        search=search
    )


@admin_bp.route("/bookings/status/<int:booking_id>", methods=["POST"])
@admin_required
def booking_status_update(booking_id):
    status = request.form.get("status", "").strip()
    if update_booking_status(booking_id, status):
        status_labels = {
            "pending": "Chờ duyệt",
            "confirmed": "Đã xác nhận",
            "completed": "Hoàn thành",
            "cancelled": "Đã hủy"
        }
        flash(f"Đã cập nhật trạng thái đơn #{booking_id} thành '{status_labels.get(status, status)}'!", "success")
    else:
        flash("Cập nhật trạng thái đơn đặt tour thất bại.", "danger")
    return redirect(url_for("admin.bookings_management"))


@admin_bp.route("/bookings/delete/<int:booking_id>", methods=["POST"])
@admin_required
def booking_delete(booking_id):
    if delete_booking(booking_id):
        flash(f"Đã xóa đơn đặt tour #{booking_id} thành công!", "success")
    else:
        flash("Xóa đơn đặt tour thất bại.", "danger")
    return redirect(url_for("admin.bookings_management"))


# =======================================================
# 8. QUẢN LÝ ĐÁNH GIÁ (REVIEWS)
# =======================================================
@admin_bp.route("/reviews")
@admin_required
def reviews_management():
    reviews_list = get_all_reviews(limit=100)
    return render_template("admin/reviews.html", reviews=reviews_list)


@admin_bp.route("/reviews/delete/<int:review_id>", methods=["POST"])
@admin_required
def review_delete(review_id):
    if delete_review(review_id):
        flash("Đã xóa đánh giá thành công!", "success")
    else:
        flash("Xóa đánh giá thất bại.", "danger")
    return redirect(url_for("admin.reviews_management"))


# =======================================================
# 9. DEEP LEARNING CONTINUAL LEARNING (TỰ HỌC TỪ LỊCH SỬ CHAT)
# =======================================================
@admin_bp.route("/self-learning")
@admin_required
def self_learning_view():
    stats = self_learning_engine.get_learning_stats()
    scan_results = self_learning_engine.scan_candidates(min_confidence=0.80, limit=200)
    return render_template(
        "admin/self_learning.html",
        stats=stats,
        scan_results=scan_results
    )


@admin_bp.route("/self-learning/run", methods=["POST"])
@admin_required
def self_learning_run():
    min_confidence = request.form.get("min_confidence", type=float) or 0.80
    result = self_learning_engine.execute_learning(min_confidence=min_confidence)
    if result.get("success"):
        flash(result.get("message", "Đã kích hoạt AI tự học thành công!"), "success")
    else:
        flash(result.get("message", "Không thể thực hiện tự học."), "danger")
    return redirect(url_for("admin.self_learning_view"))
