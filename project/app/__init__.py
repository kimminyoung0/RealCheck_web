from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate 
from datetime import timedelta
from dotenv import load_dotenv
import os

app = Flask(__name__)

# PostgreSQL 연결 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://min0:kim980317@db/database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = "secret"
app.config['DEBUG'] = True
app.config['PROPAGATE_EXCEPTIONS'] = True 
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=2)  # 세션 2시간 유지

load_dotenv()  # .env 파일 로드
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "fallback-secret-key")

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# 블루프린트 등록 (API 모듈화)
from app.auth import auth_bp
from app.predict import predict_bp
from app.routes import routes_bp

app.register_blueprint(auth_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(routes_bp)
