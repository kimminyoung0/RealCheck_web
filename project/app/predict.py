from flask import Blueprint, request, jsonify, render_template
import jwt
from app import app, db
from app.models import Input, Prediction
import pickle
import pandas as pd
import numpy as np

predict_bp = Blueprint('predict', __name__)

# 모델 로드
try:
    model = pickle.load(open("model/model_0.83.pkl", "rb"))
except FileNotFoundError:
    raise RuntimeError("❌ 모델 파일을 찾을 수 없습니다. `model/model_0.81.pkl` 확인 필요.")

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

        # 수치형 컬럼만 추출
        numeric_cols = [col for col in sample.select_dtypes(include=['number']).columns]
        sample[numeric_cols] = sample[numeric_cols].fillna(train_medians)

        # 스케일러 및 KNN 모델 로드
        scaler = pickle.load(open("./saved/scaler.pkl", "rb"))
        knn_hc = pickle.load(open("./saved/knn_hc.pkl", "rb"))

        X_test = sample[['전용면적', '방수', '욕실수']]
        if X_test.isnull().values.any():
            raise ValueError("❌ 예측에 필요한 필수 컬럼에 결측치가 있습니다.")

        X_test_scaled = scaler.transform(X_test)
        sample["매물_HC"] = knn_hc.predict(X_test_scaled)

        # 금액 단위 변환
        # sample["보증금"] = sample["보증금"] / 10000
        # sample["월세"] = sample["월세"] / 10000
        sample['월세+관리비'] = sample['월세'] + sample['관리비']
        sample['보증금_월세관리비_비율'] = sample['월세+관리비'] / sample['보증금']
        sample['전용면적_가격_비율'] = sample['보증금_월세관리비_비율'] / sample['전용면적']

        scaler2 = pickle.load(open("./saved/scaler2.pkl", "rb"))
        knn_kmedoids = pickle.load(open("./saved/knn_kmedoids.pkl", "rb"))

        X_test2 = sample[['매물_HC', '전용면적_가격_비율', '보증금_월세관리비_비율']]
        X_test_scaled2 = scaler2.transform(X_test2)
        sample["지역_KMedoids"] = knn_kmedoids.predict(X_test_scaled2)

        sample['게재일'] = pd.to_datetime(sample['게재일'], errors='coerce')
        sample['계절'] = sample['게재일'].dt.month.apply(get_season)

        date_max = pickle.load(open("./saved/date_max.pkl", "rb"))
        sample['매물_등록_경과일'] = (date_max - sample['게재일']).dt.days

        sample = pd.get_dummies(sample, columns=['매물확인방식', '방향', '주차가능여부', '계절'], drop_first=True)

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
    ##############################################
    user_id, error_response = get_user_id_from_token()
    if error_response:
        return error_response
    
    # ✅ 파일 업로드 체크 로직 추가
    if "file" not in request.files:
        return jsonify({"error": "파일이 업로드되지 않았습니다."}), 400

    file = request.files.get("file")
    
    if file is None:
        return jsonify({"error": "파일이 업로드되지 않았습니다."}), 400

    if file.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "CSV 파일만 업로드할 수 있습니다."}), 400
    
    try:
        df = pd.read_csv(file)

        # 데이터 전처리 수행
        preprocessed_df = preprocess(df)

        predictions = model.predict(preprocessed_df)
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
