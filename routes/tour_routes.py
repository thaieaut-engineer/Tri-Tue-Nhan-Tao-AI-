from flask import Blueprint, render_template, request
from models.tour import get_all_tours, get_tour_by_id, get_tour_schedule, search_tours
from models.category import get_all_categories

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

    return render_template(
        "tours.html",
        tours=tours_list,
        categories=categories,
        keyword=keyword,
        selected_category=category_id,
        min_price=min_price,
        max_price=max_price,
        sort_by=sort_by
    )


@tour_bp.route("/tours/<int:tour_id>")
def tour_detail(tour_id):
    """
    Trang chi tiết tour và lịch trình tham quan.
    """
    tour = get_tour_by_id(tour_id)
    if not tour:
        return render_template("404.html", message="Không tìm thấy tour du lịch yêu cầu"), 404

    schedule = get_tour_schedule(tour_id)
    return render_template(
        "tour_detail.html",
        tour=tour,
        schedule=schedule
    )