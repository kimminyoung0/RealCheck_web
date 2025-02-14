from flask import Blueprint, request, jsonify, render_template
import jwt
from app import app, db
from app.models import Input, Prediction
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import os

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
    
def preprocess(df):
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
        
        try:
            print("🔍 X_test 데이터 샘플:\n", X_test.head())  # X_test 원본 데이터 확인
            print("🔍 X_test 데이터 형태:", X_test.shape)  # X_test 차원 확인
            print("🔍 결측값 확인 (X_test):", X_test.isnull().sum().sum())  # 결측값 개수 확인
            print("🔍 무한대값 확인 (X_test):", np.isinf(X_test).sum().sum())  # 무한대값 개수 확인

            print("🔍 스케일러 객체 타입:", type(scaler))
            print("🔍 스케일러 정보:", scaler)  # 스케일러가 올바르게 로드되었는지 확인

            X_test_scaled = scaler.transform(X_test)  # 스케일링 수행

            print("✅ X_test_scaled 샘플:\n", X_test_scaled[:5])  # 스케일링 후 일부 데이터 출력
            print("✅ X_test_scaled 데이터 형태:", X_test_scaled.shape)  # 변환 후 차원 확인
            print("✅ KNN 모델이 기대하는 입력 차원:", knn_hc._fit_X.shape)  # 학습된 KNN 모델 차원 확인

            # KNN 예측 수행
            sample["매물_HC"] = knn_hc.predict(X_test_scaled)
            print("🎯 KNN 예측 완료! 첫 번째 예측 값:", sample["매물_HC"].iloc[0])
            
        except ValueError as ve:
            raise ValueError(f"🚨 KNN 예측 중 오류 발생 (ValueError): {str(ve)}")
        except AttributeError as ae:
            raise ValueError(f"🚨 KNN 예측 중 오류 발생 (AttributeError): {str(ae)}")
        except Exception as e:
            raise ValueError(f"🚨 KNN 예측 중 오류 발생: {str(e)}")

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
        sample = sample.drop(columns = ['ID', '중개사무소', '제공플랫폼', '게재일', '매물_DBSCAN', '월세+관리비', '보증금_월세관리비_비율'], axis = 1)
        sample = pd.get_dummies(sample, columns=['매물_HC', '지역_KMedoids'], drop_first=True)
        one_hot_columns = [col for col in sample.columns if 'HC' in col or 'KMedoids' in col]
        sample[one_hot_columns] = sample[one_hot_columns].astype(int)
        return sample
    except Exception as e:
        raise ValueError(f"🚨 데이터 전처리 중 오류 발생: {str(e)}")

def make_prediction(input_data):
    try:
        df = pd.DataFrame([input_data])
        preprocessed_df = preprocess(df)
        prediction = model.predict(preprocessed_df)
        pred_proba = model.predict_proba(preprocessed_df)
        correct_probs = pred_proba[np.arange(len(prediction)), prediction]
        percent_probs_mean = (correct_probs * 100).round(1).mean()
        return prediction, percent_probs_mean
    except Exception as e:
        raise ValueError(f"🚨 예측 중 오류 발생: {str(e)}")

#predict url로 POST 요청이 들어오면 predict()메서드를 수행하겠다는 의미
@predict_bp.route("/predict", methods=["POST"])
def predict():
    """ JSON 입력을 받아서 단일 예측 수행 """
    user_id, error_response = get_user_id_from_token()
    if error_response:
        return error_response  # 인증 실패 시 에러 반환
    
    data = request.json

    try:
        prediction_result, percent_probs_mean = make_prediction(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    new_input = Input(user_id=user_id, input_data=data)
    db.session.add(new_input)
    db.session.commit()

    new_prediction = Prediction(input_id=new_input.id, 
                                prediction_result=prediction_result, 
                                percent_probs_mean=percent_probs_mean)
    db.session.add(new_prediction)
    db.session.commit()

    return jsonify({"prediction": prediction_result, "pred_proba": percent_probs_mean})

@predict_bp.route("/predict/file", methods=["POST"])
def predict_file():
    """ CSV 파일을 업로드하여 다중 예측 수행 """
    user_id, error_response = get_user_id_from_token()
    if error_response:
        return error_response
    print("🔍 서버에서 받은 파일 목록:", request.files)
    file = request.files.get("file")
    print("📂 업로드된 파일:", file)
    ####################이 위까지만 실행됨
    if file is None:
        return jsonify({"error": "파일이 업로드되지 않았습니다."}), 400

    if file.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "CSV 파일만 업로드할 수 있습니다."}), 400
    print("여기까지 수행됨")
    try:
        df = pd.read_csv(file)
        print("df 데이터 프레임 생성!!!!!!!!!")
        # 데이터 전처리 수행
        preprocessed_df = preprocess(df)
        print("df 데이터 전처리 완료 !!!!!!!!!")
        predictions = model.predict(preprocessed_df)
        print("df 모델 예측 완료 !!!!!!!!!")
        pred_proba = model.predict_proba(preprocessed_df)
        correct_probs = pred_proba[np.arange(len(predictions)), predictions]
        confidence_scores = (correct_probs * 100).round(1)

        prediction_labels = ["허위매물이 아닙니다" if pred == 0 else "허위매물입니다" for pred in predictions]

        result_df = df.copy()
        result_df["예측 결과"] = prediction_labels
        result_df["신뢰도 (%)"] = confidence_scores

        result_html = result_df.to_html(classes="table table-striped", index=False)

        return render_template("result.html", table=result_html)

    except Exception as e:
        return jsonify({"error": "파일 예측 실패", "message": str(e)}), 400
