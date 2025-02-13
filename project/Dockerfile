# 1️⃣ Python 3.11 이미지 사용 (Slim 버전으로 가벼운 이미지 사용)
FROM python:3.11-slim

# 2️⃣ 작업 디렉토리 생성
WORKDIR /app

# 3️⃣ 필요한 패키지 설치 (PostgreSQL 개발 도구 포함)
RUN apt-get update && apt-get install -y \
    libpq-dev gcc

# 3️⃣ Python 환경 변수 설정 (버퍼링 비활성화 및 UTF-8 설정)
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PIP_NO_CACHE_DIR=1

# 4️⃣ 필요한 패키지만 우선 복사하여 캐싱 최적화
COPY requirements.txt .

# 5️⃣ 패키지 설치 (pip 업그레이드 포함)
RUN pip install --upgrade pip && pip install -r requirements.txt

# 6️⃣ 나머지 프로젝트 파일 복사 (이 단계가 늦을수록 Docker 빌드 속도 최적화)
COPY . .

# 7️⃣ Flask 실행
CMD ["python", "run.py"]