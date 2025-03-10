# 1️⃣ Python 3.11 이미지 사용 (Slim 버전으로 가벼운 이미지 사용)
FROM python:3.11-slim

# 2️⃣ 작업 디렉토리 생성
WORKDIR /app

# 3️⃣ 필요한 패키지 설치 (PostgreSQL 개발 도구 포함 + net-tools 추가)
RUN apt-get update && apt-get install -y \
    libpq-dev gcc net-tools && \
    rm -rf /var/lib/apt/lists/*  # 설치 후 불필요한 캐시 삭제하여 이미지 최적화

# 4️⃣ Python 환경 변수 설정 (버퍼링 비활성화 및 UTF-8 설정)
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PIP_NO_CACHE_DIR=1

# 5️⃣ 필요한 패키지만 우선 복사하여 캐싱 최적화
COPY requirements.txt .

# 6️⃣ 패키지 설치 (pip 업그레이드 포함)
RUN pip install --upgrade pip && pip install -r requirements.txt

# 7️⃣ 나머지 프로젝트 파일 복사 (이 단계가 늦을수록 Docker 빌드 속도 최적화)
COPY . .

# 8️⃣ Flask 실행
#CMD ["python", "run.py"]

# 배포를 위해 Elastic Beanstalk에서는 Werkzeug(Flask와 같은 Python 웹 프레임워크에서 HTTP 요청을 처리하고 WSGI 표준을 구현하는 도구)
# 를 지원하지 않기 때문에 Gunicorn으로 변경하고 4개의 워커 프로세스를 실행하도록 설정
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]

