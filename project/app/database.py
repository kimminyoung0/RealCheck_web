from app import app, db
from app.models import Users, Input, Prediction  # 🔥 Users 테이블을 먼저 가져옴

def create_tables():
    """ 🔹 Users 테이블을 먼저 생성한 후, Input 및 Prediction 테이블 생성 """
    with app.app_context():
        # db.create_all()
        print("📌 Users 테이블을 먼저 생성합니다...")
        db.metadata.create_all(bind=db.engine, tables=[Users.__table__])  # ✅ Users 테이블만 먼저 생성

        print("📌 나머지 테이블(Input, Prediction) 생성 시작...")
        db.metadata.create_all(bind=db.engine, tables=[Input.__table__, Prediction.__table__])  # ✅ 그다음 Input, Prediction 생성

        print("✅ 모든 테이블이 정상적으로 생성되었습니다!")

if __name__ == "__main__":
    create_tables()
