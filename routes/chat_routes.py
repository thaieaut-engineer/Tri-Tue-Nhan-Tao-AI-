from flask import Blueprint, render_template, request, jsonify

from ai.chatbot import Chatbot


chat_bp = Blueprint(
    "chat",
    __name__
)


# Khởi tạo chatbot
chatbot = Chatbot()


@chat_bp.route("/chatbot")
def chatbot_page():

    return render_template(
        "chatbot.html"
    )


@chat_bp.route("/api/chat", methods=["POST"])
def chat():

    data = request.get_json()

    question = data.get(
        "question",
        ""
    )

    answer = chatbot.generate_response(
        question
    )

    return jsonify({
        "answer": answer
    })