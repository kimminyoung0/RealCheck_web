from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# PostgreSQL 연결 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://min0:kim980317@db/database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = "secret"

db = SQLAlchemy(app)

# 블루프린트 등록 (API 모듈화)
from app.auth import auth_bp
from app.predict import predict_bp
from app.routes import routes_bp

app.register_blueprint(auth_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(routes_bp)
