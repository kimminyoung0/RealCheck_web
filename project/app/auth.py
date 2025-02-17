from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from app import app, db
from app.models import User

auth_bp = Blueprint('auth', __name__)

# 회원가입 API
@auth_bp.route('/register', methods=['POST'])
def register():
    """ 회원가입 API (중복 이메일 체크 포함) """
    data = request.json
    email = data.get("email")
    password = data.get("password")
    next_page = data.get("next", "/")

    # 중복 이메일 검사
    if User.query.filter_by(email=email).first():
        return jsonify({"message": "이미 가입된 이메일입니다."}), 400

    hashed_password = generate_password_hash(password)  # 비밀번호 해싱
    new_user = User(email=email, password=hashed_password)

    db.session.add(new_user)
    db.session.commit()
    
    print("✅ 회원가입 완료, 다음 페이지로 이동:", next_page)  # 🔥 로그 찍기

    return jsonify({"message": "회원가입 성공!", "next": next_page}), 201

# 로그인 API
@auth_bp.route('/login', methods=['POST'])
def login():
    """ 사용자 로그인 API (JWT 토큰 발급) """
    data = request.json
    email = data.get("email")
    password = data.get("password")
    next_page = data.get("next", "/")  # 기본적으로 홈으로 이동

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        return jsonify({"message": "이메일 또는 비밀번호가 올바르지 않습니다."}), 401

    token = jwt.encode(
        {
            "user_id": user.id,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)  # 2시간 유효
        },
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )

    return jsonify({"token": token})
