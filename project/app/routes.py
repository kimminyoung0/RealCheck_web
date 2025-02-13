from flask import Blueprint, jsonify, render_template, redirect, url_for, session

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

@app.route("/mypage")
def mypage():
    """ My Page 화면 """
    user_id, error_response = get_user_id_from_token()

    # 🔹 비회원이면 로그인 페이지로 리디렉트
    if user_id is None:
        return redirect(url_for("login"))

    # 🔹 사용자 정보 가져오기 (세션에서)
    user_email = session.get("user_email", "이메일 정보 없음")
    user_join_date = session.get("user_join_date", "가입일 정보 없음")

    # 🔹 예측 이력 가져오기
    predictions = get_user_predictions(user_id)

    return render_template("mypage.html", user_email=user_email, user_join_date=user_join_date, predictions=predictions)
