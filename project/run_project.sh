#!/bin/bash

echo "🚀 Docker 컨테이너 재빌드 시작..."
docker compose down
docker compose build --no-cache

echo "🚀 Docker 컨테이너 실행 중..."
docker compose up -d

sleep 5

echo "🚀 컨테이너 내부로 이동하여 Python 실행..."
docker exec -it project-web-1 bash -c "export PYTHONPATH=/app && python -m run"

