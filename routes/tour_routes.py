from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from routes.auth_routes import login_required
from models.tour import get_all_tours, get_tour_by_id, get_tour_schedule, search_tours
from models.category import get_all_categories
from models.booking import create_booking, get_user_bookings, cancel_booking_by_user
from models.review import create_review, get_reviews_by_tour, get_tour_rating_stats
from models.favorite import toggle_favorite, is_tour_favorite, get_user_favorites, get_user_favorite_tour_ids
from models.user import get_user_by_id

tour_bp = Blueprint("tour", __name__)


@tour_bp.route("/tours")
def tours():
    """
    Trang danh sách tour du lịch, hỗ trợ tìm kiếm theo từ khóa, lọc theo danh mục, giá và sắp xếp.
    """
    keyword = request.args.get("q", "").strip()
    category_id = request.args.get("category_id", type=int)
    min_price = request.args.get("min_price", type=float)
    max_price = request.args.get("max_price", type=float)
    sort_by = request.args.get("sort", "").strip()

    categories = get_all_categories()

    # Nếu có bất kỳ điều kiện lọc nào, gọi hàm search_tours
    if keyword or category_id or min_price is not None or max_price is not None or sort_by:
        tours_list = search_tours(
            keyword=keyword if keyword else None,
            category_id=category_id if category_id else None,
            min_price=min_price,
            max_price=max_price,
            sort_by=sort_by if sort_by else None
        )
    else:
        tours_list = get_all_tours()

    # Gắn điểm đánh giá trung bình cho mỗi tour
    for t in tours_list:
        stats = get_tour_rating_stats(t["id"])
        t["avg_rating"] = stats["avg_rating"]
        t["total_reviews"] = stats["total_reviews"]

    favorite_ids = get_user_favorite_tour_ids(session.get("user_id"))

    return render_template(
        "tours.html",
        tours=tours_list,
        categories=categories,
        keyword=keyword,
        selected_category=category_id,
        min_price=min_price,
        max_price=max_price,
        sort_by=sort_by,
        favorite_ids=favorite_ids
    )


@tour_bp.route("/tours/<int:tour_id>")
def tour_detail(tour_id):
    """
    Trang chi tiết tour và lịch trình tham quan, kèm đánh giá và đặt tour.
    """
    tour = get_tour_by_id(tour_id)
    if not tour:
        return render_template("404.html", message="Không tìm thấy tour du lịch yêu cầu"), 404

    schedule = get_tour_schedule(tour_id)
    reviews = get_reviews_by_tour(tour_id)
    rating_stats = get_tour_rating_stats(tour_id)
    is_favorited = is_tour_favorite(session.get("user_id"), tour_id)

    # Thông tin người dùng hiện tại nếu đã đăng nhập để điền trước vào form đặt tour
    current_user = None
    if "user_id" in session:
        current_user = get_user_by_id(session["user_id"])

    return render_template(
        "tour_detail.html",
        tour=tour,
        schedule=schedule,
        reviews=reviews,
        rating_stats=rating_stats,
        is_favorited=is_favorited,
        current_user=current_user
    )


@tour_bp.route("/tours/<int:tour_id>/book", methods=["POST"])
def book_tour(tour_id):
    """
    Xử lý gửi đơn đặt tour du lịch.
    """
    tour = get_tour_by_id(tour_id)
    if not tour:
        flash("Tour du lịch không tồn tại.", "danger")
        return redirect(url_for("tour.tours"))

    full_name = request.form.get("full_name", "").strip()
    phone = request.form.get("phone", "").strip()
    email = request.form.get("email", "").strip()
    start_date = request.form.get("start_date", "").strip()
    adults = request.form.get("adults", type=int) or 1
    children = request.form.get("children", type=int) or 0
    note = request.form.get("note", "").strip()

    if not full_name or not phone or not email or not start_date:
        flash("Vui lòng điền đầy đủ họ tên, số điện thoại, email và ngày khởi hành.", "danger")
        return redirect(url_for("tour.tour_detail", tour_id=tour_id))

    if adults < 1:
        flash("Số lượng người lớn tối thiểu là 1.", "danger")
        return redirect(url_for("tour.tour_detail", tour_id=tour_id))

    # Tính tổng giá tour: người lớn 100%, trẻ em (5-9 tuổi) 75%
    base_price = float(tour["price"])
    total_price = (adults * base_price) + (children * base_price * 0.75)

    user_id = session.get("user_id")

    booking_id = create_booking(
        user_id=user_id,
        tour_id=tour_id,
        full_name=full_name,
        phone=phone,
        email=email,
        start_date=start_date,
        adults=adults,
        children=children,
        total_price=total_price,
        note=note
    )

    if booking_id:
        flash(f"🎉 Đặt tour thành công! Mã đơn đặt tour: #{booking_id}. Nhân viên sẽ sớm liên hệ xác nhận với bạn.", "success")
        if user_id:
            return redirect(url_for("tour.my_bookings"))
        return redirect(url_for("tour.tour_detail", tour_id=tour_id))
    else:
        flash("Có lỗi xảy ra khi tạo đơn đặt tour. Vui lòng thử lại sau.", "danger")
        return redirect(url_for("tour.tour_detail", tour_id=tour_id))


@tour_bp.route("/tours/<int:tour_id>/review", methods=["POST"])
@login_required
def submit_review(tour_id):
    """
    Gửi đánh giá và bình luận cho tour.
    """
    tour = get_tour_by_id(tour_id)
    if not tour:
        flash("Tour không tồn tại.", "danger")
        return redirect(url_for("tour.tours"))

    rating = request.form.get("rating", type=int)
    comment = request.form.get("comment", "").strip()

    if not rating or not (1 <= rating <= 5):
        flash("Vui lòng chọn số sao đánh giá hợp lệ (1-5 sao).", "warning")
        return redirect(url_for("tour.tour_detail", tour_id=tour_id))

    if not comment:
        flash("Vui lòng nhập nội dung đánh giá của bạn.", "warning")
        return redirect(url_for("tour.tour_detail", tour_id=tour_id))

    success, msg = create_review(
        user_id=session["user_id"],
        tour_id=tour_id,
        rating=rating,
        comment=comment
    )

    if success:
        flash("Cảm ơn bạn đã gửi đánh giá cho tour này!", "success")
    else:
        flash(f"Lỗi khi gửi đánh giá: {msg}", "danger")

    return redirect(url_for("tour.tour_detail", tour_id=tour_id))


@tour_bp.route("/api/favorites/toggle", methods=["POST"])
def api_toggle_favorite():
    """
    API AJAX thêm / xóa tour yêu thích.
    """
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Vui lòng đăng nhập để lưu tour yêu thích.", "need_login": True}), 401

    data = request.get_json() or {}
    tour_id = data.get("tour_id") or request.form.get("tour_id", type=int)

    if not tour_id:
        return jsonify({"success": False, "message": "Mã tour không hợp lệ."}), 400

    is_favorited, message = toggle_favorite(session["user_id"], tour_id)
    return jsonify({
        "success": True,
        "is_favorited": is_favorited,
        "message": message
    })


@tour_bp.route("/favorites")
@login_required
def favorites():
    """
    Trang danh sách các tour du lịch mà người dùng đã lưu yêu thích.
    """
    fav_tours = get_user_favorites(session["user_id"])
    for t in fav_tours:
        stats = get_tour_rating_stats(t["id"])
        t["avg_rating"] = stats["avg_rating"]
        t["total_reviews"] = stats["total_reviews"]

    return render_template("favorites.html", tours=fav_tours)


@tour_bp.route("/my-bookings")
@login_required
def my_bookings():
    """
    Trang lịch sử đơn đặt tour của người dùng.
    """
    bookings_list = get_user_bookings(session["user_id"])
    return render_template("my_bookings.html", bookings=bookings_list)


@tour_bp.route("/my-bookings/cancel/<int:booking_id>", methods=["POST"])
@login_required
def cancel_booking(booking_id):
    """
    Người dùng hủy đơn đặt tour đang chờ duyệt.
    """
    success = cancel_booking_by_user(booking_id, session["user_id"])
    if success:
        flash("Đã hủy đơn đặt tour thành công.", "success")
    else:
        flash("Không thể hủy đơn đặt tour này (đơn đã được xác nhận hoặc không tồn tại).", "danger")

    return redirect(url_for("tour.my_bookings"))