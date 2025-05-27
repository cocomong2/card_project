import pandas as pd

def add_outlier_flag(df, column):
    """
    주어진 컬럼에 대해 IQR 기반 이상치 여부 플래그 컬럼을 생성합니다.
    이상치는 제거하지 않고 플래그로만 표시합니다.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    flag_col = f"{column}_outlier"
    df[flag_col] = ((df[column] < lower) | (df[column] > upper)).astype(int)
    return df

def preprocess_vip_data(data):
    """
    VIP 등급 예측용 전처리 함수:
    - 파생변수 생성
    - 불필요 컬럼 제거
    - 이상치 플래그 추가
    """
    try:
        # 파생변수 생성
        data['이용건수_B0M'] = data[['이용건수_일시불_B0M','이용건수_체크_B0M','이용건수_할부_B0M']].sum(axis=1)
        data['이용금액_B0M'] = data[['이용금액_일시불_B0M','이용금액_체크_B0M','이용금액_할부_B0M']].sum(axis=1)
        data['최종이용일자'] = data[['최종이용일자_일시불','최종이용일자_체크','최종이용일자_할부']].max(axis=1)
        data['이용후경과월'] = data[['이용후경과월_일시불','이용후경과월_체크','이용후경과월_할부']].sum(axis=1)
        data['이용건수_R12M'] = data[['이용건수_일시불_R12M','이용건수_체크_R12M','이용건수_할부_R12M']].sum(axis=1)
        data['이용금액_R12M'] = data[['이용금액_일시불_R12M','이용금액_체크_R12M','이용금액_할부_R12M']].sum(axis=1)
        data['최대이용금액_R12M'] = data[['최대이용금액_일시불_R12M','최대이용금액_체크_R12M','최대이용금액_할부_R12M']].sum(axis=1)
        data['이용개월수_R12M'] = data[['이용개월수_일시불_R12M','이용개월수_체크_R12M','이용개월수_할부_R12M']].sum(axis=1)
        data['이용건수_R6M'] = data[['이용건수_일시불_R6M','이용건수_체크_R6M','이용건수_할부_R6M']].sum(axis=1)
        data['이용금액_R6M'] = data[['이용금액_일시불_R6M','이용금액_체크_R6M','이용금액_할부_R6M']].sum(axis=1)
        data['이용개월수_R6M'] = data[['이용개월수_일시불_R6M','이용개월수_체크_R6M','이용개월수_할부_R6M']].sum(axis=1)
        data['이용건수_R3M'] = data[['이용건수_일시불_R3M','이용건수_체크_R3M','이용건수_할부_R3M']].sum(axis=1)
        data['이용금액_R3M'] = data[['이용금액_일시불_R3M','이용금액_체크_R3M','이용금액_할부_R3M']].sum(axis=1)
        data['이용개월수_R3M'] = data[['이용개월수_일시불_R3M','이용개월수_체크_R3M','이용개월수_할부_R3M']].sum(axis=1)
        data['회원여부_이용가능'] = data[['회원여부_이용가능','회원여부_이용가능_CA','회원여부_이용가능_카드론']].sum(axis=1)
        data['유효카드수'] = data[['유효카드수_신용체크','유효카드수_신용','유효카드수_체크']].sum(axis=1)
        data['이용금액_증감률'] = (data['이용금액_R3M'] - data['이용금액_R6M']) / (data['이용금액_R6M'] + 1)
        data['최근_이용금액_비중'] = data['이용금액_B0M'] / (data['이용금액_R12M'] + 1)
        data['VIP등급코드'] = data['VIP등급코드'].str.replace("_", "0").astype(int)
        data['이용금액대'] = data['이용금액대'].str.extract(r'^(\\d+)').astype(int)
        data['연령'] = data['연령'].str.replace("대|이상", "", regex=True).astype(int)

        # 이상치 플래그
        data = add_outlier_flag(data, '이용금액_R12M')
        data = add_outlier_flag(data, '이용금액_증감률')

        # 필요 없는 컬럼 제거
        data.drop(columns=[col for col in data.columns if col.startswith((
            '이용건수_일시불', '이용건수_체크', '이용건수_할부',
            '이용금액_일시불', '이용금액_체크', '이용금액_할부',
            '최종이용일자_', '이용후경과월_',
            '최대이용금액_', '이용개월수_', '회원여부_이용가능', '유효카드수_', '발급회원번호'
        ))], errors='ignore', inplace=True)

    except Exception as e:
        print(f"[ERROR] 전처리 중 오류 발생: {e}")
    return data

    if __name__ == "__main__":
        import pandas as pd

# 예시 데이터프레임
df = pd.DataFrame({
    '이용금액_일시불_B0M': [10000], '이용금액_체크_B0M': [5000], '이용금액_할부_B0M': [2000],
    '이용금액_일시불_R12M': [200000], '이용금액_체크_R12M': [100000], '이용금액_할부_R12M': [50000],
    '이용금액_일시불_R6M': [80000], '이용금액_체크_R6M': [30000], '이용금액_할부_R6M': [10000],
    '이용금액_일시불_R3M': [60000], '이용금액_체크_R3M': [20000], '이용금액_할부_R3M': [5000],
    'VIP등급코드': ['6'], '이용금액대': ['3천만원대'], '연령': ['30대'],
    '회원여부_이용가능': [1], '회원여부_이용가능_CA': [0], '회원여부_이용가능_카드론': [0],
    '유효카드수_신용체크': [1], '유효카드수_신용': [1], '유효카드수_체크': [0],
    '이용건수_일시불_B0M': [1], '이용건수_체크_B0M': [2], '이용건수_할부_B0M': [0],
    '이용건수_일시불_R12M': [10], '이용건수_체크_R12M': [5], '이용건수_할부_R12M': [2],
    '이용개월수_일시불_R12M': [6], '이용개월수_체크_R12M': [6], '이용개월수_할부_R12M': [4],
    '최종이용일자_일시불': [15], '최종이용일자_체크': [20], '최종이용일자_할부': [18],
    '이용후경과월_일시불': [2], '이용후경과월_체크': [3], '이용후경과월_할부': [1],
})

result = preprocess_vip_data(df)
print(result.head())
