from preprocessing import preprocessing_data
import os
import joblib
import pymysql
import pandas as pd


conn = pymysql.connect(
    host='localhost',
    user='wms',
    password='1234',
    db='card',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,"final_vip_model.pkl")

model = joblib.load(MODEL_PATH)

def predict_from_input(input_list):
    """
    input_dict: 예측에 사용할 하나의 샘플 데이터 (딕셔너리)
    return: 예측 결과와 클래스 1 확률
    """
    # 입력을 DataFrame으로 변환
    df = pd.DataFrame(input_list)
 
    # 전처리 수행
    df_processed = preprocessing_data(df)
    
    # 
    proba = model.predict_proba(df_processed)
    
    pred = model.predict(df_processed)
    
    proba_for_pred = [proba[i][pred[i]] for i in range(len(pred))]
 
 
    return pred, proba_for_pred
 
with conn.cursor() as cursor:
    sql = "SELECT * FROM card_leaver"  # VIP등업? 데이터베이스 이름 넣기
    cursor.execute(sql)
    row = cursor.fetchall()  # dict로 반환됨
 
    if row:
        predict_from_input(row)
    else:
        print("데이터가 없습니다.")