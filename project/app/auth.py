from flask import Blueprint, request, redirect, jsonify, session
from app import db
from app.models import Users
import requests

auth_bp = Blueprint('auth', __name__)

KAKAO_TOKEN_URL = "https://kauth.kakao.com/oauth/token"
KAKAO_USER_URL = "https://kapi.kakao.com/v2/user/me"
CLIENT_ID = "b33f3e54487184ca0a1f259a2cd1eb1d"
REDIRECT_URI = "http://127.0.0.1:6000/auth/kakao/callback"

@auth_bp.route("/auth/kakao/login")
def kakao_login():
    """ 카카오 로그인 요청 (카카오 로그인 페이지로 리다이렉트) """
    kakao_auth_url = (
        f"https://kauth.kakao.com/oauth/authorize"
        f"?client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
    )
    return redirect(kakao_auth_url)

@auth_bp.route("/auth/kakao/callback")
def kakao_callback():
    """ 카카오 로그인 콜백 """
    code = request.args.get("code")
    if not code:
        return jsonify({"message": "인가 코드 없음"}), 400

    # 1️⃣ 카카오에서 액세스 토큰 요청
    token_data = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "code": code
    }
    token_res = requests.post(KAKAO_TOKEN_URL, data=token_data).json()
    access_token = token_res.get("access_token")

    if not access_token:
        return jsonify({"message": "토큰 요청 실패"}), 400

    # 2️⃣ 액세스 토큰으로 사용자 정보 요청
    headers = {"Authorization": f"Bearer {access_token}"}
    user_res = requests.get(KAKAO_USER_URL, headers=headers).json()

    kakao_id = user_res["id"]
    email = user_res["kakao_account"].get("email", f"{kakao_id}@kakao.com")
    #nickname = user_res["properties"]["nickname"]
    
    # 3️⃣ DB에 사용자 저장 (이미 있으면 패스)
    user = Users.query.filter_by(email=email).first()
    if not user:
        user = Users(email=email, kakao_id=kakao_id)
        db.session.add(user)
        db.session.commit()

    # 세션 저장 (로그인 유지)
    session.permanent = True # 세션을 지속적으로 유지하도록 설정
    session["user_id"] = user.id
    session["email"] = user.email
    
    return jsonify({"message": "카카오 로그인 성공!", "email": email}), 200

@auth_bp.route("/auth/logout")
def logout():
    """ 로그아웃 (세션 삭제) """
    session.clear()
    return jsonify({"message": "로그아웃 완료!"}), 200