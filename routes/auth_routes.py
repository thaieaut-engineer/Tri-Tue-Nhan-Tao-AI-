from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from models.user import (
    verify_user, create_user, get_user_by_username, get_user_by_id,
    update_user_profile, change_user_password
)
from models.booking import get_user_bookings
from models.favorite import get_user_favorites

auth_bp = Blueprint("auth", __name__)


def login_required(f):
    """
    Decorator kiểm tra đăng nhập trước khi truy cập route.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Vui lòng đăng nhập để tiếp tục.", "warning")
            return redirect(url_for("auth.login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """
    Decorator kiểm tra quyền Admin trước khi truy cập route quản trị.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Vui lòng đăng nhập với tài khoản Quản trị viên.", "warning")
            return redirect(url_for("auth.login", next=request.url))
        if session.get("role") != "admin":
            flash("Bạn không có quyền truy cập khu vực quản trị.", "danger")
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated_function


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """
    Trang đăng ký tài khoản người dùng mới.
    """
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        full_name = request.form.get("full_name", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not username or not full_name or not password:
            flash("Vui lòng điền đầy đủ các thông tin bắt buộc.", "danger")
            return render_template("register.html", username=username, full_name=full_name)

        if len(username) < 3:
            flash("Tên đăng nhập phải có ít nhất 3 ký tự.", "danger")
            return render_template("register.html", username=username, full_name=full_name)

        if len(password) < 6:
            flash("Mật khẩu phải có độ dài tối thiểu 6 ký tự.", "danger")
            return render_template("register.html", username=username, full_name=full_name)

        if password != confirm_password:
            flash("Mật khẩu xác nhận không trùng khớp.", "danger")
            return render_template("register.html", username=username, full_name=full_name)

        success, message = create_user(username=username, password=password, full_name=full_name, role="user")
        if success:
            flash("Đăng ký tài khoản thành công! Mời bạn đăng nhập.", "success")
            return redirect(url_for("auth.login"))
        else:
            flash(f"Đăng ký thất bại: {message}", "danger")
            return render_template("register.html", username=username, full_name=full_name)

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """
    Trang đăng nhập hệ thống.
    """
    if "user_id" in session:
        if session.get("role") == "admin":
            return redirect(url_for("admin.dashboard"))
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Vui lòng nhập tên đăng nhập và mật khẩu.", "danger")
            return render_template("login.html", username=username)

        user = verify_user(username, password)
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["full_name"] = user["full_name"]
            session["role"] = user["role"]

            flash(f"Xin chào mừng trở lại, {user['full_name']}!", "success")

            next_page = request.args.get("next")
            if next_page:
                return redirect(next_page)

            if user["role"] == "admin":
                return redirect(url_for("admin.dashboard"))
            return redirect(url_for("home"))
        else:
            flash("Tên đăng nhập hoặc mật khẩu không chính xác.", "danger")
            return render_template("login.html", username=username)

    return render_template("login.html")


@auth_bp.route("/logout")
def logout():
    """
    Đăng xuất khỏi hệ thống.
    """
    session.clear()
    flash("Đã đăng xuất thành công.", "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """
    Trang quản lý thông tin cá nhân và đổi mật khẩu người dùng.
    """
    user_id = session["user_id"]

    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "update_profile":
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip()
            phone = request.form.get("phone", "").strip()

            if not full_name:
                flash("Họ và tên không được để trống.", "danger")
            else:
                success, msg = update_user_profile(user_id, full_name, email, phone)
                if success:
                    session["full_name"] = full_name
                    flash(msg, "success")
                else:
                    flash(msg, "danger")

        elif action == "change_password":
            old_password = request.form.get("old_password", "").strip()
            new_password = request.form.get("new_password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()

            if not old_password or not new_password:
                flash("Vui lòng nhập đầy đủ mật khẩu hiện tại và mật khẩu mới.", "danger")
            elif new_password != confirm_password:
                flash("Mật khẩu mới xác nhận không khớp.", "danger")
            elif len(new_password) < 6:
                flash("Mật khẩu mới phải có ít nhất 6 ký tự.", "danger")
            else:
                success, msg = change_user_password(user_id, old_password, new_password)
                if success:
                    flash(msg, "success")
                else:
                    flash(msg, "danger")

        return redirect(url_for("auth.profile"))

    user = get_user_by_id(user_id)
    bookings = get_user_bookings(user_id)
    favorites = get_user_favorites(user_id)

    return render_template(
        "profile.html",
        user=user,
        booking_count=len(bookings),
        favorite_count=len(favorites)
    )
