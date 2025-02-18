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
        #X_test_scaled = scaler.transform(X_test.values.reshape(1, -1) if len(X_test.shape) == 1 else X_test)
        print(f"✅ X_test_scaled shape: {X_test_scaled.shape}")
        
        sample["매물_HC"] = knn_hc.predict(X_test_scaled)

        sample["매물_DBSCAN"] = knn_dbscan.predict(X_test_scaled)
        
        print(f"✅ 매물_DBSCAN shape: {sample['매물_DBSCAN'].shape}")
        
        # 금액 단위 변환
        # sample["보증금"] = sample["보증금"] / 10000
        # sample["월세"] = sample["월세"] / 10000
        
        sample['월세+관리비'] = sample['월세'] + sample['관리비']
        sample['보증금_월세관리비_비율'] = sample['월세+관리비'] / sample['보증금']
        sample['전용면적_가격_비율'] = sample['보증금_월세관리비_비율'] / sample['전용면적']
        print("1111111")
        
        scaler2 = pickle.load(open("./saved/scaler2.pkl", "rb"))
        knn_kmedoids = pickle.load(open("./saved/knn_kmedoids.pkl", "rb"))
        print("2222222")
        
        X_test2 = sample[['매물_HC', '매물_DBSCAN', '전용면적_가격_비율', '보증금_월세관리비_비율']]
        #X_test_scaled2 = scaler2.transform(X_test2)
        X_test_scaled2= scaler.transform(X_test2.values.reshape(1, -1) if len(X_test2.shape) == 1 else X_test2)

        print(f"✅ X_test_scaled2 shape: {X_test_scaled2.shape}")
        
        #X_test_scaled2 = scaler2.transform(np.atleast_2d(X_test2))
        #X_test_scaled2 = pd.DataFrame(X_test_scaled2, columns=X_test2.columns, index=X_test2.index)
        print("3333333")
        sample["지역_KMedoids"] = knn_kmedoids.predict(X_test_scaled2)
        #sample["지역_KMedoids"] = np.array(knn_kmedoids.predict(X_test_scaled2)).reshape(-1)
        print(f"✅ 지역_KMedoids shape: {sample['지역_KMedoids'].shape}")
        print("4444444")
        sample['게재일'] = pd.to_datetime(sample['게재일'], errors='coerce')
        sample['계절'] = sample['게재일'].dt.month.apply(get_season)
        print("55555")
        date_max = pickle.load(open("./saved/date_max.pkl", "rb"))
        sample['매물_등록_경과일'] = (date_max - sample['게재일']).dt.days
        print("666666")
        sample = pd.get_dummies(sample, columns=['매물확인방식', '방향', '주차가능여부', '계절'], drop_first=True)
        sample = sample.drop(columns = ['ID', '중개사무소', '제공플랫폼', '게재일', '매물_DBSCAN', '월세+관리비', '보증금_월세관리비_비율'], axis = 1)
        sample = pd.get_dummies(sample, columns=['매물_HC', '지역_KMedoids'], drop_first=True)
        one_hot_columns = [col for col in sample.columns if 'HC' in col or 'KMedoids' in col]
        sample[one_hot_columns] = sample[one_hot_columns].astype(int)
        print("sample data: ", sample)
        return sample
    except Exception as e:
        raise ValueError(f"🚨 데이터 전처리 중 오류 발생: {str(e)}")

def generate_random_id():
    """랜덤한 4자리 문자 + 6자리 숫자로 이루어진 ID 생성"""
    letters = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))  # 대문자 + 숫자 4자리
    numbers = ''.join(random.choices(string.digits, k=6))  # 숫자 6자리
    return f"{letters}{numbers}"


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
    """ JSON 입력을 받아 단일 예측 수행 및 DB 저장 """
    user_id, error_response = get_user_id_from_token()
    if error_response:
        return error_response  # 인증 실패 시 에러 반환

    data = request.json
    print("🔍입력 받은 data", data)
    if not data:
        return jsonify({"error": "입력 데이터가 제공되지 않았습니다."}), 400

    try:
        df = pd.DataFrame([data])  # JSON 데이터를 DataFrame으로 변환
        print("df 데이터 프레임 생성!!!!!!!!!2")
        df['ID'] = generate_random_id() ############추후에 user_id값과 랜덤숫자의조합으로 만들기
        print(df)
        # 데이터 전처리 수행
        preprocessed_df = preprocess(df)
        print("df 데이터 전처리 완료 !!!!!!!!!")
    
        if preprocessed_df.isna().sum().sum() > 0:
            print("🚨 전처리 후에도 NaN이 남아 있음")
            print(preprocessed_df.isna().sum())
            
        preprocessed_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # 무한대 값을 NaN으로 변환
        preprocessed_df.fillna('-', inplace=True)  # NaN을 0으로 변환
        print("변환완료!!!!!!!!!!!")
        print(preprocessed_df)
        # 예측 수행####################아래에서부터문제발생여기서부터해결하기
        predictions = model.predict(preprocessed_df)
        print("df 모델 예측 완료 !!!!!!!!!")

        print("이 함수가 실행되는거 맞ㅈ니........")
        try:
            print("try문 들어옴.......")
            # 예측 확률 계산
            pred_proba = model.predict_proba(preprocessed_df)
            print("pred_proba : ", pred_proba)
            # 차원 문제 해결
            predictions = np.array(predictions).flatten()  # 1차원 배열로 변환
            print("predictions : ", predictions)
            correct_probs = pred_proba[np.arange(len(predictions)), predictions]  # 안전한 인덱싱
            print("correct_probs : ", correct_probs)
            confidence_scores = (correct_probs * 100).round(1).astype(float)
            print("confidence_scores : ", confidence_scores)
            # 예측 결과 변환
            prediction_labels = ["허위매물이 아닙니다" if pred == 0 else "허위매물입니다" for pred in predictions]
        except Exception as e:
            print("🚨 예측 과정에서 오류 발생:", str(e))
            import traceback
            traceback.print_exc()  # 상세한 오류 로그 출력
            return "예측 중 오류 발생: " + str(e), 400 
        
        try:
            # DB에 입력 데이터 저장
            df = df.where(pd.notna(df), None)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)  # 무한대 값을 NaN으로 변환
            df.fillna('-', inplace=True)  # NaN을 0으로 변환

            json_data = json.dumps(df.to_dict(orient="records"), allow_nan = False)
            new_input = Input(user_id=user_id if user_id is not None else None, input_data=json_data)

            db.session.add(new_input)
            db.session.commit()

            # DB에 예측 결과 저장
            for pred, conf in zip(prediction_labels, confidence_scores):
                new_prediction = Prediction(input_id=new_input.id, 
                                            prediction_result=pred, 
                                            confidence=conf)
                db.session.add(new_prediction)

            db.session.commit()
        except Exception as e:
            print("🚨 DB 저장 과정에서 오류 발생:", str(e))
            import traceback
            traceback.print_exc()  # 상세한 오류 로그 출력
            return "DB 저장 중 오류 발생: " + str(e), 400 
        

        result_df = df.copy()
        result_df["예측 결과"] = prediction_labels
        result_df["신뢰도 (%)"] = confidence_scores

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
        print("df 데이터 프레임 생성1")

        # 데이터 전처리 수행
        preprocessed_df = preprocess(df)
        print("df 데이터 전처리 완료 !!!!!!!!!")
        
        if preprocessed_df.isna().sum().sum() > 0:
            print("🚨 전처리 후에도 NaN이 남아 있음")
            print(preprocessed_df.isna().sum())
        
        preprocessed_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # 무한대 값을 NaN으로 변환
        preprocessed_df.fillna(0, inplace=True)  # NaN을 0으로 변환

        print(preprocessed_df)
        # 예측 수행
        predictions = model.predict(preprocessed_df)
        print("df 모델 예측 완료 !!!!!!!!!")

        # 예측 확률 계산
        pred_proba = model.predict_proba(preprocessed_df)
        correct_probs = pred_proba[np.arange(len(predictions)), predictions]
        confidence_scores = (correct_probs * 100).round(1).astype(float)

        # 예측 결과 변환
        prediction_labels = ["허위매물이 아닙니다" if pred == 0 else "허위매물입니다" for pred in predictions]

        # DB에 입력 데이터 저장
        df = df.where(pd.notna(df), None)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)  # 무한대 값을 NaN으로 변환
        df.fillna('-', inplace=True)  # NaN을 0으로 변환

        json_data = json.dumps(df.to_dict(orient="records"), allow_nan = False)
        new_input = Input(user_id=user_id if user_id is not None else None, input_data=json_data)

        db.session.add(new_input)
        db.session.commit()

        # DB에 예측 결과 저장
        for pred, conf in zip(prediction_labels, confidence_scores):
            new_prediction = Prediction(input_id=new_input.id, 
                                        prediction_result=pred, 
                                        confidence=conf)
            db.session.add(new_prediction)

        db.session.commit()

        result_df = df.copy()
        result_df["예측 결과"] = prediction_labels
        result_df["신뢰도 (%)"] = confidence_scores

        result_html = result_df.to_html(classes="table table-striped", index=False)
        print("predict.py의 predict_file() 메서드 모두 완료")

        return render_template("result.html", table=result_html)

    except Exception as e:
        return jsonify({"error": "파일 예측 실패", "message": str(e)}), 400
