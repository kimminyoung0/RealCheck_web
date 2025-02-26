from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from app import app, db
from app.models import Users

auth_bp = Blueprint('auth', __name__)

# 회원가입 API
@auth_bp.route('/register', methods=['POST'])
def register():
    """ 회원가입 API (중복 이메일 체크 포함) """
    data = request.json
    data = request.get_json()
    if not data:
        return jsonify({"message": "잘못된 요청입니다. JSON 데이터를 보내주세요."}), 400
    email = data.get("email")
    password = data.get("password")
    next_page = data.get("next", "/")

    # 중복 이메일 검사
    if Users.query.filter_by(email=email).first():
        print(f"🔍 회원가입 시도 - 이미 가입된 이메일 : {email}")
        return jsonify({"message": "이미 가입된 이메일입니다."}), 400

    hashed_password = generate_password_hash(password)  # 비밀번호 해싱
    print(f"✅ 생성된 해시: {hashed_password}")
    new_user = Users(email=email, password=hashed_password)

    try:
        db.session.add(new_user)
        db.session.commit()
    except Exception as e:
        db.session.rollback() #트랜잭션 롤백해서 트랜잭션을 깨끗하게 정리
        print(f"❌ user 데이터 저장 실패: {e}")
    finally:
        db.session.close() 

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

    user = Users.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        print(f"❌ 비밀번호 검증 실패: 입력된 비밀번호: {password}, 저장된 해시: {user.password}")
        return jsonify({"message": "이메일 또는 비밀번호가 올바르지 않습니다."}), 401

    token = jwt.encode(
        {
            "user_id": user.id,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)  # 2시간 유효
        },
        app.config["SECRET"],
        algorithm="HS256"
    )
    print(f"✅ 토큰 발급 성공: {token}")  # 🔥 로그 추가
    return jsonify({"token": token})
