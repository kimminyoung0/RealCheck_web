from flask import Blueprint, request, render_template, session, redirect, jsonify
from app.models import Users
from app import db

my_page_bp = Blueprint("my_page", __name__)

@my_page_bp.route("/mypage", methods=["GET"])
def mypage():
    """ 마이페이지 렌더링 """
    if "user_id" not in session:
        return redirect("/auth/kakao/login")

    user = Users.query.get(session["user_id"])
    if not user:
        return redirect("/auth/kakao/login")

    return render_template("mypage.html", user=user)

@my_page_bp.route("/mypage/data", methods=["GET"])
def mypage_data():
    """ 마이페이지 - 사용자 정보 API """
    if "user_id" not in session:
        return jsonify({"message": "로그인이 필요합니다."}), 401

    user = Users.query.get(session["user_id"])
    if not user:
        return jsonify({"message": "사용자 정보를 찾을 수 없습니다."}), 404

    return jsonify({
        "user_id": user.id,
        "nickname": user.profile_nickname,
        "profile_image": user.profile_image
    }), 200
