from flask import Flask

from routes.tour_routes import tour_bp
from routes.chat_routes import chat_bp


app = Flask(__name__)

app.config["SECRET_KEY"] = "chatbot_tour_secret_key"


# =========================
# REGISTER BLUEPRINT
# =========================

app.register_blueprint(tour_bp)

app.register_blueprint(chat_bp)


# =========================
# HOME
# =========================

@app.route("/")
def home():

    return """
        <h1>🤖 Chatbot tư vấn tour du lịch</h1>

        <p>Hệ thống đang hoạt động!</p>

        <br>

        <a href="/tours">
            🌏 Xem danh sách tour
        </a>

        <br><br>

        <a href="/chatbot">
            🤖 Chat với chatbot
        </a>
    """


if __name__ == "__main__":

    app.run(
        debug=True
    )