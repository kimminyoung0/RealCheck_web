# RealCheck_web
부동산 허위매물 데이터를 활용해 해당 매물이 허위매물인지 예측하는 경진대회에 참여해 모델링을 진행했습니다.
현재는 모델 고도화를 위해 사용자 관련 데이터와 실매물 데이터를 수집하기 위한 웹서비스를 기획 및 구현했습니다.

# 부동산 허위매물 예측 웹 서비스
--------------------------------------
## 🛠️ 기술 스택
- **언어**: Python, JavaScript, SQL
- **라이브러리**: Scikit-learn, XGBoost
- **데이터 분석 & 처리**: Pandas, NumPy
- **MLOps & 배포**: Docker, Flask  
- **데이터베이스**: PostgreSQL
---------------------------------
## 블로그 정리 및 이슈 해결 사항
XGBoost를 선택한 이유
[https://velog.io/@kimminyoung0/ML-XGBoost-LightGBM-CatBoost-비교](https://velog.io/@kimminyoung0/ML-XGBoost-LightGBM-CatBoost-비교)

- [매물을 저층, 중층, 고층으로 나누고 싶을 때](https://velog.io/@kimminyoung0/매물을-저층-중층-고층으로-나누고-싶을-때-t-분포f-분포)
- [데이터 전처리](https://velog.io/@kimminyoung0/데이터-전처리)

---------------------------------
## Dataset Info
train.csv
ID : 샘플별 고유 ID / 매물확인방식 / 보증금 / 월세 / 전용면적 / 해당층 / 총층 / 방향 / 방수 / 욕실수 / 주차가능여부 / 총주차대수 / 관리비 / 중개사무소 / 제공플랫폼 / 게재일 / 허위매물여부

test.csv 
ID : 샘플별 고유 ID / 매물확인방식 / 보증금 / 월세 / 전용면적 / 해당층 / 총층 / 방향 / 방수 / 욕실수 / 주차가능여부 / 총주차대수 / 관리비 / 중개사무소 / 제공플랫폼 / 게재일

sample_submission.csv - 제출 양식
ID : 샘플별 고유 ID / 허위매물여부

------------------------------------------

## EDA
[EDA 블로그 정리](https://velog.io/@kimminyoung0/EDA)

## DataPreprocessing
### 허위매물 판단 기준
- 주변 시세보다 가격이 싼지
- 매물을 등록한지 오래되었는지
- 다른 매물에 비해 기본 정보가 누락되었는지
  
위와 같은 허위매물 판단 기준으로 새로운 변수들을 생성하고 EDA 분석 결과에 따라 feature engineering 진행했습니다.

### 매물 클러스터링 및 지역 클러스터링
- 매물은 전용면적, 방수, 욕실수등의 변수를 사용해 계층적 군집화와 DBSCAN으로 클러스터링을 진행했습니다.
- 지역 클러스터링은 매물_HC, 매물_DBSCAN, 전용면적_가격_비율, 보증금_월세관리비_비율 등의 변수를 사용해 KMedoids와 HDBSCAN으로 클러스터링을 진행했습니다.
![지역별매물산점도_전용면적_보증금월세관리비비율_통합](https://github.com/user-attachments/assets/44b4e02d-74f8-4eb5-8cdf-0e8ddf1ba933)
![지역별매물산점도_보증금_월세관리비_통합](https://github.com/user-attachments/assets/619071dd-8c89-4959-bc28-66090e2c6c3b)

- 실험을 통해 모델 성능이 좋은 매물_HC와 지역_KMedoids를 이용해 훈련 데이터에 포함시켰습니다.
### 매물 등록 경과일 변수 생성
훈련 데이터의 게재일 변수 최대값과 해당 샘플의 게재일 차이를 이용해 매물 등록 경과일 변수를 생성했습니다.
### 결측치 개수 변수 생성
하나의 샘플에서 결측치 개수를 계산해 변수로 생성했습니다.

### 다중공선성 완화
다중공선성을 완화하기 위해 히트맵으로 상관관계를 분석하고 SHAP 분석을 진행한 후 VIF 계수도 함께 검토하여 제거할 변수를 선택했습니다. 

## Model Train & Predict
XGBoost를 사용했습니다.

### 모델 성능 평가

## 데이터베이스 구조 설계

## 모델 배포 (Deployment)
- Flask 및 Docker를 활용한 API 개발
- PostgreSQL 연동 및 데이터 저장 기능 구현
- 데이터 조회 기능 개발중

## 웹 서비스 개발 (Web Service)
- Flask 기반의 백엔드 구축
- Docker를 이용한 컨테이너화 및 배포
<img width="2265" alt="스크린샷 2025-02-19 오전 4 47 39" src="https://github.com/user-attachments/assets/085c7747-bd80-4be8-8bd3-cf92168d0cc4" />


