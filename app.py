import os
from flask import Flask, render_template
from dotenv import load_dotenv

from routes.tour_routes import tour_bp
from routes.chat_routes import chat_bp
from routes.auth_routes import auth_bp
from routes.admin_routes import admin_bp
from models.tour import get_all_tours

# Tải cấu hình từ .env
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "chatbot_tour_secret_key")


# =======================================================
# ĐĂNG KÝ CÁC BLUEPRINT
# =======================================================
app.register_blueprint(tour_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)


# =======================================================
# TRANG CHỦ
# =======================================================
@app.route("/")
def home():
    """
    Trang chủ giới thiệu hệ thống và danh sách tour nổi bật.
    """
    tours = get_all_tours()
    featured_tours = tours[:6] if tours else []
    return render_template("index.html", tours=featured_tours)


# =======================================================
# XỬ LÝ LỖI (ERROR HANDLERS)
# =======================================================
@app.errorhandler(404)
def not_found_error(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )