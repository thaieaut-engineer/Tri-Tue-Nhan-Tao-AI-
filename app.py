from flask import Flask

from routes.tour_routes import tour_bp


app = Flask(__name__)

app.config["SECRET_KEY"] = "chatbot_tour_secret_key"


# Đăng ký route Tour
app.register_blueprint(tour_bp)


@app.route("/")
def home():
    return """
        <h1>Chatbot tư vấn tour du lịch</h1>

        <p>Hệ thống đang hoạt động!</p>

        <a href="/tours">Xem danh sách tour</a>
    """


if __name__ == "__main__":
    app.run(
        debug=True
    )