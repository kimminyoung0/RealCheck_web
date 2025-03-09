from app import db
from werkzeug.security import generate_password_hash
from sqlalchemy.dialects.postgresql import ARRAY

class Users(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    def __init__(self, email, password):
        self.email = email
        self.password = generate_password_hash(password)  # 비밀번호 암호화 저장
        
class Input(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    input_data = db.Column(db.JSON, nullable=False)  # JSON으로 입력 데이터 저장
    uploaded_file = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_id = db.Column(db.Integer, db.ForeignKey('input.id'), nullable=False)
    prediction_result = db.Column(ARRAY(db.String), nullable=False)  # ✅ 리스트로 저장 가능
    confidence = db.Column(ARRAY(db.Float), nullable=True)  # 리스트로 저장 가능
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
