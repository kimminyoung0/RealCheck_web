from flask import Blueprint, request, redirect, jsonify, session, url_for
from app import db
from app.models import Users
import requests

auth_bp = Blueprint('auth', __name__)

KAKAO_TOKEN_URL = "https://kauth.kakao.com/oauth/token"
KAKAO_USER_URL = "https://kapi.kakao.com/v2/user/me"
CLIENT_ID = "b33f3e54487184ca0a1f259a2cd1eb1d"
REDIRECT_URI = "http://localhost:6010/auth/kakao/callback"

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
    print("카카오 응답 데이터:", user_res)
    
    kakao_id = user_res["id"]
    profile_data = user_res.get("kakao_account", {}).get("profile", {})
    profile_nickname = profile_data.get("nickname", "사용자")
    profile_image = profile_data.get("profile_image_url", None)
    print("profile_image url 길이", len(profile_image))
    
    # 3️⃣ DB에 사용자 저장
    user = Users.query.get(kakao_id)
    if not user:
        user = Users(id=kakao_id, profile_nickname=profile_nickname, profile_image=profile_image)
        db.session.add(user)
        db.session.commit()

    # 세션 저장 (로그인 유지)
    session.permanent = True # 세션을 지속적으로 유지하도록 설정
    session["user_id"] = user.id
    session["profile_nickname"] = user.profile_nickname
    session["profile_image"] = user.profile_image
    
    return redirect(url_for('routes.index'))

@auth_bp.route("/auth/logout")
def logout():
    """ 로그아웃 (세션 삭제 후 홈으로 이동) """
    session.clear()
    return redirect(url_for("routes.index"))  # 로그아웃 후 홈으로 이동



# @auth_bp.route("/auth/status")
# def auth_status():
#     """ 로그인 상태 확인 """
#     user_id = session.get("user_id")
#     profile_nickname = session.get("profile_nickname")

#     print("auth.py 파일의 auth_status 실행 중!")

#     # JavaScript에서 요청한 경우 JSON 응답 반환
#     if request.headers.get("X-Requested-With") == "XMLHttpRequest":
#         if user_id:
#             return jsonify({"user": {"id": user_id, "nickname": profile_nickname}}), 200
#         else:
#             return jsonify({"user": None}), 200

#     # 브라우저에서 직접 접근한 경우 로그인 페이지로 이동
#     return redirect(url_for("auth.kakao_login"))

@auth_bp.route("/auth/status")
def auth_status():
    """ 로그인 상태 확인 """
    
    user_id = session.get("user_id")
    profile_nickname = session.get("profile_nickname")

    print("auth.py 파일의 auth_status 실행 중!")

    # 🔹 항상 JSON 반환 (리다이렉트 제거)
    return jsonify({
        "user": {"id": user_id, "nickname": profile_nickname} if user_id else None
    }), 200
