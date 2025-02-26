from app import app, db
from app.models import Users, Input, Prediction

# 데이터베이스를 초기화하고 테이블을 생성하는 역할
# 서버 실행 전에 한 번 실행해서 테이블을 생성해야 함.
# DB 테이블 생성 함수
def create_tables():
    with app.app_context():
        db.create_all()
        print("✅ Database tables created!")

if __name__ == "__main__":
    create_tables()
