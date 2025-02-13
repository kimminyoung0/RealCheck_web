from flask import Blueprint, jsonify, render_template

# Flask 앱의 엔드포인트를 관리하는 역할을 함
# auth.py와 predict.py에 있는 API를 중앙에서 관리하도록 도와줌
routes_bp = Blueprint("routes", __name__)

@routes_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@routes_bp.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")

@routes_bp.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")

@routes_bp.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")

@routes_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "running"})


