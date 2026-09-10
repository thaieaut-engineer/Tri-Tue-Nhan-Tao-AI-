from flask import Blueprint, render_template

from models.tour import get_all_tours
from models.tour import get_tour_by_id
from models.tour import get_tour_schedule


tour_bp = Blueprint("tour", __name__)


@tour_bp.route("/tours")
def tours():
    """
    Trang danh sách tour.
    """

    tours = get_all_tours()

    return render_template(
        "tours.html",
        tours=tours
    )


@tour_bp.route("/tours/<int:tour_id>")
def tour_detail(tour_id):
    """
    Trang chi tiết tour.
    """

    tour = get_tour_by_id(tour_id)

    if not tour:
        return "Không tìm thấy tour", 404

    schedule = get_tour_schedule(tour_id)

    return render_template(
        "tour_detail.html",
        tour=tour,
        schedule=schedule
    )