from flask import Blueprint, request, jsonify, render_template
import jwt
from app import app, db
from app.models import Input, Prediction
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
import random
import string

predict_bp = Blueprint('predict', __name__)

# 모델 로드
try:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # `app/`의 상위 폴더로 이동
    MODEL_PATH = os.path.join(BASE_DIR, "model", "final_model.json")  # 모델 경로 설정

    # 🔹 모델 로드
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH) 
except FileNotFoundError:
    raise RuntimeError("❌ 모델 파일을 찾을 수 없습니다.")

def get_user_id_from_token():
    """ JWT 토큰에서 user_id 추출 """
    token = request.headers.get("Authorization")
    if not token:
        return None, None  # 비회원이면 None 반환
    try:
        decoded = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        return decoded["user_id"], None  # 로그인한 회원이면 user_id 반환
    except jwt.ExpiredSignatureError:
        return None, jsonify({"message": "Token expired"})  # 토큰 만료
    except jwt.InvalidTokenError:
        return None, jsonify({"message": "Invalid token"})  # 잘못된 토큰

def get_season(month):
    """ 데이터 전처리 중 일부 """
    if month in [3, 4, 5]:
        return '봄'
    elif month in [6, 7, 8]:
        return '여름'
    elif month in [9, 10, 11]:
        return '가을'
    else:
        return '겨울'
    
def preprocess_for_file(df):
    try:
        sample = df.copy()
        sample2 = df.copy()

        if sample.empty:
            raise ValueError("❌ 입력 데이터가 비어 있습니다.")

        # 결측 여부를 이진 변수로 변환
        for col in sample2.columns:
            sample2[f'{col}'] = sample2[col].isna().astype(int)
        sample2['결측치개수'] = sample2.sum(axis=1)
        sample['결측치개수'] = sample2['결측치개수']
        
        train_medians = pd.read_csv('./saved/train_medians.csv')
        train_medians_dict = train_medians.set_index("column_name")["median_value"].to_dict()  # train_medians가 DataFrame이라면 변환
        
        # 수치형 컬럼만 추출
        numeric_cols = [col for col in sample.select_dtypes(include=['number']).columns]
        sample[numeric_cols] = sample[numeric_cols].fillna(sample[numeric_cols].apply(lambda col: train_medians_dict.get(col.name, col.median())))
        
        # 스케일러 및 KNN 모델 로드
        scaler = pickle.load(open("./saved/scaler.pkl", "rb"))
        knn_hc = pickle.load(open("./saved/knn_hc.pkl", "rb"))
        knn_dbscan = pickle.load(open("./saved/knn_dbscan.pkl", "rb"))
        X_test = sample[['전용면적', '방수', '욕실수']]
        X_test_scaled = scaler.transform(X_test)

        # KNN 예측 수행
        sample["매물_HC"] = knn_hc.predict(X_test_scaled)
        sample["매물_DBSCAN"] = knn_dbscan.predict(X_test_scaled)
        
        # 금액 단위 변환
        # sample["보증금"] = sample["보증금"] / 10000
        # sample["월세"] = sample["월세"] / 10000
        
        sample['월세+관리비'] = sample['월세'] + sample['관리비']
        sample['보증금_월세관리비_비율'] = sample['월세+관리비'] / sample['보증금']
        sample['전용면적_가격_비율'] = sample['보증금_월세관리비_비율'] / sample['전용면적']
        
        scaler2 = pickle.load(open("./saved/scaler2.pkl", "rb"))
        knn_kmedoids = pickle.load(open("./saved/knn_kmedoids.pkl", "rb"))
        
        X_test2 = sample[['매물_HC', '매물_DBSCAN', '전용면적_가격_비율', '보증금_월세관리비_비율']]
        X_test_scaled2 = scaler2.transform(X_test2)
        
        sample["지역_KMedoids"] = knn_kmedoids.predict(X_test_scaled2)
    
        sample['게재일'] = pd.to_datetime(sample['게재일'], errors='coerce')
        sample['계절'] = sample['게재일'].dt.month.apply(get_season)
        
        date_max = pickle.load(open("./saved/date_max.pkl", "rb"))
        sample['매물_등록_경과일'] = (date_max - sample['게재일']).dt.days
        
        sample = pd.get_dummies(sample, columns=['매물확인방식', '방향', '주차가능여부', '계절'], drop_first=True)
        one_hot_columns = [col for col in sample.columns if '매물확인방식' in col or '방향' in col or '주차가능여부' in col or '계절' in col]
        sample[one_hot_columns] = sample[one_hot_columns].astype(int)
        
        sample = sample.drop(columns = ['ID', '중개사무소', '제공플랫폼', '게재일', '매물_DBSCAN', '월세+관리비', '보증금_월세관리비_비율'], axis = 1)
        sample = pd.get_dummies(sample, columns=['매물_HC', '지역_KMedoids'], drop_first=True)
        one_hot_columns = [col for col in sample.columns if 'HC' in col or 'KMedoids' in col]
        sample[one_hot_columns] = sample[one_hot_columns].astype(int)
        return sample
    except Exception as e:
        raise ValueError(f"🚨 데이터 전처리 중 오류 발생: {str(e)}")
    
def preprocess_for_one(df):
    try:
        sample = df.copy()
        sample2 = df.copy()

        if sample.empty:
            raise ValueError("❌ 입력 데이터가 비어 있습니다.")

        # 결측 여부를 이진 변수로 변환
        for col in sample2.columns:
            sample2[f'{col}'] = sample2[col].isna().astype(int)
        sample2['결측치개수'] = sample2.sum(axis=1)
        sample['결측치개수'] = sample2['결측치개수']
        
        train_medians = pd.read_csv('./saved/train_medians.csv')
        train_medians_dict = train_medians.set_index("column_name")["median_value"].to_dict()  # train_medians가 DataFrame이라면 변환
        
        # 수치형 컬럼만 추출
        numeric_cols = [col for col in sample.select_dtypes(include=['number']).columns]
        sample[numeric_cols] = sample[numeric_cols].fillna(sample[numeric_cols].apply(lambda col: train_medians_dict.get(col.name, col.median())))
        
        # 스케일러 및 KNN 모델 로드
        scaler = pickle.load(open("./saved/scaler.pkl", "rb"))
        knn_hc = pickle.load(open("./saved/knn_hc.pkl", "rb"))
        knn_dbscan = pickle.load(open("./saved/knn_dbscan.pkl", "rb"))
        X_test = sample[['전용면적', '방수', '욕실수']]
        X_test_scaled = scaler.transform(X_test)

        X_test_scaled = scaler.transform(X_test)  # 스케일링 수행

        # KNN 예측 수행
        sample["매물_HC"] = knn_hc.predict(X_test_scaled)
        sample["매물_DBSCAN"] = knn_dbscan.predict(X_test_scaled)

        # 금액 단위 변환
        # sample["보증금"] = sample["보증금"] / 10000
        # sample["월세"] = sample["월세"] / 10000
        
        sample['월세+관리비'] = sample['월세'] + sample['관리비']
        sample['보증금_월세관리비_비율'] = sample['월세+관리비'] / sample['보증금']
        sample['전용면적_가격_비율'] = sample['보증금_월세관리비_비율'] / sample['전용면적']
        
        scaler2 = pickle.load(open("./saved/scaler2.pkl", "rb"))
        knn_kmedoids = pickle.load(open("./saved/knn_kmedoids.pkl", "rb"))
        
        X_test2 = sample[['매물_HC', '매물_DBSCAN', '전용면적_가격_비율', '보증금_월세관리비_비율']]
        X_test_scaled2 = scaler2.transform(X_test2)
        
        sample["지역_KMedoids"] = knn_kmedoids.predict(X_test_scaled2)
        sample['게재일'] = pd.to_datetime(sample['게재일'], errors='coerce')
        sample['계절'] = sample['게재일'].dt.month.apply(get_season)
        
        date_max = pickle.load(open("./saved/date_max.pkl", "rb"))
        sample['매물_등록_경과일'] = (date_max - sample['게재일']).dt.days
        
        # 원-핫 인코딩을 적용할 컬럼 및 제외할 값
        one_hot_columns = {
            "매물확인방식": ["현장확인", "전화확인"],  # "서류확인"은 생성하지 않음
            "방향": ["서향", "동향", "남향", "북동향", "북향", "남서향", "북서향"],  # "남동향"은 생성하지 않음
            "주차가능여부": ["불가능"],  # "가능"은 생성하지 않음
            "계절": ["봄", "여름", "겨울"]  # "가을"은 생성하지 않음
        }

        # 원-핫 인코딩 적용 (drop_first=True 효과 적용)
        for col, categories in one_hot_columns.items():
            for cat in categories:
                sample[f"{col}_{cat}"] = (sample[col] == cat).astype(int)

        # 기존 카테고리 컬럼 삭제
        sample = sample.drop(columns=one_hot_columns.keys(), errors="ignore")
        sample = sample.drop(columns = ['ID', '중개사무소', '제공플랫폼', '게재일', '매물_DBSCAN', '월세+관리비', '보증금_월세관리비_비율'], axis = 1)
        
        for i in range(2, 7): 
            sample[f"매물_HC_{i}"] = 0
        
        for i in range(1, 11):
            sample[f"지역_KMedoids_{i}"] = 0
        
        hc_value = sample['매물_HC'].iloc[0]  # 첫 번째 row의 값
        if hc_value != 1:  # 원핫인코딩으로 인해 매물_HC_1 컬럼 생성 x
            sample[f"매물_HC_{hc_value}"] = 1
        
        km_value = sample['지역_KMedoids'].iloc[0]  # 첫 번째 row의 값
        if km_value != 0:  # 원핫인코딩으로 인해 지역_KMedoids_0 컬럼 생성 x
            sample[f"지역_KMedoids_{km_value}"] = 1
            
        sample = sample.drop(columns = ['매물_HC', '지역_KMedoids'], axis = 1)
        print("최종 컬럼 수 : ", len(sample.columns))
        print("최종 컬럼들 : ", sample.columns)
        return sample
    except Exception as e:
        raise ValueError(f"🚨 데이터 전처리 중 오류 발생: {str(e)}")

def generate_random_id():
    """랜덤한 4자리 문자 + 6자리 숫자로 이루어진 ID 생성"""
    letters = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))  # 대문자 + 숫자 4자리
    numbers = ''.join(random.choices(string.digits, k=6))  # 숫자 6자리
    return f"{letters}{numbers}"


#predict url로 POST 요청이 들어오면 predict()메서드를 수행하겠다는 의미
@predict_bp.route("/input/one", methods=["POST"])
def input_one():
    """ 단일 예측 입력 → DB 저장 → 예측 실행 """
    user_id, error_response = get_user_id_from_token()
    if error_response:
        return error_response  

    data = {
        "매물확인방식": request.form.get("매물확인방식"),
        "월세": float(request.form.get("월세", 0)),
        "보증금": float(request.form.get("보증금", 0)),
        "관리비": float(request.form.get("관리비", 0)),
        "전용면적": float(request.form.get("전용면적", 0)),
        "방수": int(request.form.get("방수", 0)),
        "욕실수": int(request.form.get("욕실수", 0)),
        "방향": request.form.get("방향"),
        "해당층": int(request.form.get("해당층", 0)),
        "총층": int(request.form.get("총층", 0)),
        "총주차대수": int(request.form.get("총주차대수", 0)),
        "주차가능여부": request.form.get("주차가능여부"),
        "제공플랫폼": request.form.get("제공플랫폼"),
        "중개사무소": request.form.get("중개사무소"),
        "게재일": request.form.get("게재일") + " 00:00:00" if request.form.get("게재일") else None
    }

    df = pd.DataFrame([data])
    df['ID'] = generate_random_id()
    df.insert(0, 'ID', df.pop('ID'))
    
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df.fillna('0', inplace=True)  # NaN 값을 0으로 대체
    missing_mask = df.isna()  # NaN 위치 저장
    df = df.astype(object)  # 모든 데이터를 object 타입으로 변환 (숫자도 포함)
    df[missing_mask] = "-"  # NaN이었던 곳만 "-"로 변경

    try:
        json_data = json.dumps(df.to_dict(orient="records"), ensure_ascii=False, allow_nan=False)
        new_input = Input(user_id=user_id, input_data=json_data)

        db.session.add(new_input)
        db.session.commit()
        db.session.refresh(new_input)

        return predict_from_db(new_input.id)  # ✅ 바로 예측 실행

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "데이터 저장 실패", "message": str(e)}), 500

@predict_bp.route("/input/file", methods=["POST"])
def input_file():
    """ CSV 파일 업로드 → DB 저장 → 예측 실행 """
    user_id, error_response = get_user_id_from_token()
    if error_response:
        return error_response

    file = request.files.get("file")
    if file is None or file.filename == "" or not file.filename.endswith(".csv"):
        return jsonify({"error": "CSV 파일만 업로드할 수 있습니다."}), 400

    try:
        df = pd.read_csv(file)
        #df.replace([np.inf, -np.inf], np.nan, inplace=True)
        #df.fillna('0', inplace=True)  # NaN 값을 0으로 대체
        
        missing_mask = df.isna()  # NaN 위치 저장
        df = df.astype(object)  # 모든 데이터를 object 타입으로 변환 (숫자도 포함)
        df[missing_mask] = "-"  # NaN이었던 곳만 "-"로 변경
        
        json_data = json.dumps(df.to_dict(orient="records"), ensure_ascii=False, allow_nan=False)

        new_input = Input(user_id=user_id if user_id is not None else None, input_data=json_data)

        db.session.add(new_input)
        db.session.commit()
        db.session.refresh(new_input)

        return predict_from_db(new_input.id)  #  바로 예측 실행

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "파일 저장 실패", "message": str(e)}), 500

def predict_from_db(input_id):
    """ 데이터베이스에서 불러와 예측 수행 """
    try:
        input_record = Input.query.get(input_id)
        if not input_record:
            return jsonify({"error": "해당 ID의 데이터가 존재하지 않습니다."}), 404

        df = pd.DataFrame(json.loads(input_record.input_data))

        # "-"를 NaN으로 변환 (원래 결측치였던 값)
        df.replace("-", np.nan, inplace=True)
        
        # 단일 입력, 파일 입력 전처리 구분
        if len(df) == 1:
            preprocessed_df = preprocess_for_one(df)  # 단일 입력 처리
        else:
            preprocessed_df = preprocess_for_file(df)  #파일 입력 처리

        predictions = model.predict(preprocessed_df)
        pred_proba = model.predict_proba(preprocessed_df)
        correct_probs = pred_proba[np.arange(len(predictions)), predictions]
        confidence_scores = (correct_probs * 100).round(1).astype(float).tolist()

        prediction_labels = ["허위매물이 아닙니다" if pred == 0 else "허위매물입니다" for pred in predictions]

        new_prediction = Prediction(
            input_id=input_id,
            prediction_result=prediction_labels,
            confidence=confidence_scores
        )

        db.session.add(new_prediction)
        db.session.commit()

        result_df = df.copy()
        result_df["예측 결과"] = prediction_labels[0] if isinstance(prediction_labels, list) else prediction_labels
        result_df["신뢰도 (%)"] = confidence_scores[0] if isinstance(confidence_scores, list) else confidence_scores

        result_html = result_df.to_html(classes="table table-striped", index=False)

        return render_template("result.html", table=result_html)

    except Exception as e:
        return jsonify({"error": "예측 실패", "message": str(e)}), 400