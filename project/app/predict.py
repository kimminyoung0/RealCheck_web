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
        return None, None  # ✅ 비회원이면 None 반환 (에러 X)
    try:
        decoded = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        return decoded["user_id"], None  # ✅ 로그인한 회원이면 user_id 반환
    except jwt.ExpiredSignatureError:
        return None, jsonify({"message": "Token expired"})  # ✅ 토큰 만료
    except jwt.InvalidTokenError:
        return None, jsonify({"message": "Invalid token"})  # ✅ 잘못된 토큰

def get_season(month):
    """ 월(month)에 따라 계절 반환 """
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

        print("✅ X_test_scaled 샘플:\n", X_test_scaled[:5])  # 스케일링 후 일부 데이터 출력
        print("✅ X_test_scaled 데이터 형태:", X_test_scaled.shape)  # 변환 후 차원 확인
        print("✅ KNN 모델이 기대하는 입력 차원:", knn_hc._fit_X.shape)  # 학습된 KNN 모델 차원 확인

        # KNN 예측 수행
        sample["매물_HC"] = knn_hc.predict(X_test_scaled)
        print("🎯 KNN 예측 완료! 첫 번째 예측 값:", sample["매물_HC"].iloc[0])

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

        print("✅ X_test_scaled 샘플:\n", X_test_scaled[:5])  # 스케일링 후 일부 데이터 출력
        print("✅ X_test_scaled 데이터 형태:", X_test_scaled.shape)  # 변환 후 차원 확인
        print("✅ KNN 모델이 기대하는 입력 차원:", knn_hc._fit_X.shape)  # 학습된 KNN 모델 차원 확인

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
        print("sample.columns:", sample.columns)
        sample['게재일'] = pd.to_datetime(sample['게재일'], errors='coerce')
        sample['계절'] = sample['게재일'].dt.month.apply(get_season)
        
        date_max = pickle.load(open("./saved/date_max.pkl", "rb"))
        sample['매물_등록_경과일'] = (date_max - sample['게재일']).dt.days
        print("여기서 중간 점검 sample.매물확인방식:", sample["매물확인방식"].iloc[0])
        print("여기서 중간 점검 sample.방향:", sample["방향"].iloc[0])
        
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
        print("필요없는 column 제거 후 중간점검 :", sample.columns)
        
        for i in range(2, 7):  # 2부터 6까지 반복
            sample[f"매물_HC_{i}"] = 0
        
        for i in range(1, 11):  # 1부터 10까지 반복
            sample[f"지역_KMedoids_{i}"] = 0
        
        hc_value = sample['매물_HC'].iloc[0]  # 첫 번째 row의 값
        if hc_value != 1:  # 원핫인코딩으로 인해 매물_HC_1 컬럼 생성 x
            sample[f"매물_HC_{hc_value}"] = 1
        
        km_value = sample['지역_KMedoids'].iloc[0]  # 첫 번째 row의 값
        if km_value != 0:  # 원핫인코딩으로 인해 지역_KMedoids_0 컬럼 생성 x
            sample[f"지역_KMedoids_{km_value}"] = 1
            
        sample = sample.drop(columns = ['매물_HC', '지역_KMedoids'], axis = 1)
        print("여기서도 클러스터링 컬럼들 있는지 중간 점검, sample.columns : ", sample.columns)
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
@predict_bp.route("/predict", methods=["POST"])
def predict():
    """ 단일 예측 수행 및 DB 저장 """
    user_id, error_response = get_user_id_from_token()
    if error_response:
        return error_response  # 인증 실패 시 에러 반환
    
    print("user_id:", user_id)

    """ FormData 입력을 받아서 예측 수행 """
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


    try:
        df = pd.DataFrame([data])  # JSON 데이터를 DataFrame으로 변환
        print("✅ 단일 입력값 데이터 프레임 생성")
        df['ID'] = generate_random_id() ############추후에 user_id값과 랜덤숫자의조합으로 만들기
        print(df)
        # 데이터 전처리 수행
        preprocessed_df = preprocess_for_one(df)
        print("✅ 단일 입력값 데이터 전처리 완료")
    
        if preprocessed_df.isna().sum().sum() > 0:
            print("🚨 전처리 후에도 NaN이 남아 있음")
            print(preprocessed_df.isna().sum())
            
        preprocessed_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # 무한대 값을 NaN으로 변환
        preprocessed_df.fillna('-', inplace=True)  # NaN을 0으로 변환
        
        preprocessed_df = preprocessed_df[model.feature_names_in_]

        try:
            predictions = model.predict(preprocessed_df)
            print("Predictions:", predictions)  # 예측 결과 출력
        except Exception as e:
            print("예측 중 오류 발생:", e)
        print("📊 단일 입력값 모델 예측 완료")

        # 예측 확률 계산
        pred_proba = model.predict_proba(preprocessed_df)
        correct_probs = pred_proba[np.arange(len(predictions)), predictions]
        confidence_scores = (correct_probs * 100).round(1).astype(float).tolist()
        print("confidence_scores : ", confidence_scores)
        # 예측 결과 변환
        prediction_labels = ["허위매물이 아닙니다" if pred == 0 else "허위매물입니다" for pred in predictions]
        print("prediction_labels :", prediction_labels)
        
        # ✅ 단일 값도 리스트로 변환
        if isinstance(confidence_scores, float):  # 단일 값이면 리스트로 변환
            confidence_scores = [confidence_scores]
        if isinstance(prediction_labels, str):  # 단일 값이면 리스트로 변환
            prediction_labels = [prediction_labels]

        # DB에 입력 데이터 저장하기 전 전처리
        #df = df.where(pd.notna(df), None)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)  # 무한대 값을 NaN으로 변환
        df.fillna('-', inplace=True)  # NaN을 0으로 변환
        json_data = json.dumps(df.to_dict(orient="records"), allow_nan = False)
        
        # DB에 입력 데이터 저장
        new_input = Input(user_id=user_id if user_id is not None else None, input_data=json_data)
        #new_input = Input(user_id=user_id, input_data=data)
        
        db.session.add(new_input)
        db.session.commit()

        # DB에 예측 결과 저장
        new_prediction = Prediction(input_id=new_input.id, 
                                    prediction_result=prediction_labels, 
                                    confidence=confidence_scores)
        db.session.add(new_prediction)
        db.session.commit()

        result_df = df.copy()
        result_df["예측 결과"] = prediction_labels[0]
        result_df["신뢰도 (%)"] = confidence_scores[0]

        result_html = result_df.to_html(classes="table table-striped", index=False)
        print("predict.py의 predict() 메서드 모두 완료")

        return render_template("result.html", table=result_html)

    except Exception as e:
        return jsonify({"error": "예측 실패", "message": str(e)}), 400


@predict_bp.route("/predict/file", methods=["POST"])
def predict_file():
    """ CSV 파일을 업로드하여 다중 예측 수행 및 DB 저장 """
    
    user_id, error_response = get_user_id_from_token()
    if error_response:
        return error_response
    
    print("user_id:", user_id)
    
    print("🔍 서버에서 받은 파일 목록:", request.files)
    file = request.files.get("file")
    print("📂 업로드된 파일:", file)

    if file is None:
        return jsonify({"error": "파일이 업로드되지 않았습니다."}), 400

    if file.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "CSV 파일만 업로드할 수 있습니다."}), 400

    try:
        df = pd.read_csv(file)
        print("csv 파일 데이터 프레임 생성")

        # 데이터 전처리 수행
        preprocessed_df = preprocess_for_file(df)
        print("csv 파일 데이터 전처리 완료")
        
        if preprocessed_df.isna().sum().sum() > 0:
            print("🚨 전처리 후에도 NaN이 남아 있음")
            print(preprocessed_df.isna().sum())
        
        preprocessed_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # 무한대 값을 NaN으로 변환
        preprocessed_df.fillna(0, inplace=True)  # NaN을 0으로 변환

        print("preprocessed_df : ", preprocessed_df)
        
        # 예측 수행
        predictions = model.predict(preprocessed_df)
        print("📊 csv 파일 데이터 모델 예측 완료")
        
        # 예측 확률 계산
        pred_proba = model.predict_proba(preprocessed_df)
        correct_probs = pred_proba[np.arange(len(predictions)), predictions]
        confidence_scores = (correct_probs * 100).round(1).astype(float).tolist()  # ✅ 리스트 변환

        # 예측 결과 변환
        prediction_labels = ["허위매물이 아닙니다" if pred == 0 else "허위매물입니다" for pred in predictions]

        # ✅ JSON 변환이 필요한 경우만 json.dumps() 사용
        #prediction_labels_json = json.dumps(prediction_labels, ensure_ascii=False)
        #confidence_scores_json = json.dumps(confidence_scores)  # ✅ 리스트를 JSON 문자열로 변환

        # 원본 데이터도 JSON으로 변환하여 저장
        #df = df.where(pd.notna(df), None)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna('-', inplace=True)
        json_data = json.dumps(df.to_dict(orient="records"), ensure_ascii=False, allow_nan=False)

        # DB에 입력 데이터 저장
        new_input = Input(user_id=user_id if user_id is not None else None, input_data=json_data)
        db.session.add(new_input)
        db.session.commit()

        # ✅ DB에 예측 결과 저장 (ARRAY 타입을 지원하면 변환 없이 저장)
        new_prediction = Prediction(
            input_id=new_input.id,
            prediction_result=prediction_labels,  # ✅ JSON 컬럼이면 json.dumps() 필요
            confidence=confidence_scores  # ✅ 리스트 그대로 저장
        )
        db.session.add(new_prediction)
        db.session.commit()

        # 결과 데이터프레임 생성
        result_df = df.copy()
        result_df["예측 결과"] = prediction_labels
        result_df["신뢰도 (%)"] = confidence_scores

        # HTML로 변환
        result_html = result_df.to_html(classes="table table-striped", index=False)
        print("✅ 모든 과정 완료")

        return render_template("result.html", table=result_html)


    except Exception as e:
        return jsonify({"error": "예측 실패", "message": str(e)}), 400