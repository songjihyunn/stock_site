from datetime import datetime
import datetime
import json
import random
import mariadb
import matplotlib
import mplfinance as mpf
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import plotly.graph_objects as go
import os
import plotly.subplots as ms
import requests
import vertexai
import vertexai.preview.generative_models as generative_models
import yfinance as yf
from flask import Flask, request, render_template, jsonify
from flask import redirect
from keras import Sequential
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from tensorflow.keras.layers import LSTM, Dense
from vertexai.generative_models import GenerativeModel
from xgboost import XGBRegressor
from pdfminer.high_level import extract_text

# MySQL 연결 설정
db_config = {
    'user': 'user1',
    'password': '1111',
    'host': '192.168.0.110',
    'port': 3307,
    'database': 'pro3'
}

matplotlib.use('Agg')

app = Flask(__name__)

@app.route('/')
def indexs():
    return render_template('analyze.html')

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')


def process_data(start_date, end_date, stock_code):
    try:
        df = yf.download(stock_code, start_date, end_date)
    except Exception as e:
        error_message = f"Failed to download stock data: {str(e)}"
        return None, error_message

    if df.empty:
        return None, "No data available for the selected date range."

    # 인덱스 날짜로 설정 (mplfinance 요구사항)
    df.index.name = 'Date'
    return df, None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 폼 데이터 가져오기
        start_date = None
        end_date = None
        stock_code = None

        start_date = datetime.datetime.strptime(request.form['start_date'], '%Y-%m-%d')
        end_date = datetime.datetime.strptime(request.form['end_date'], '%Y-%m-%d')
        stock_code = request.form['stock_code']

        # 데이터 처리 및 그래프 생성
        df, error_message = process_data(start_date, end_date, stock_code)
        print(start_date)
        print(end_date)
        print(stock_code)
        if df is not None:
            # 그래프 생성
            graph_filename = 'static/candle_chart.png'  # 고정된 파일 이름 사용
            if (end_date - start_date).days > 40:
                # 이동평균선 추가
                df['MA20'] = df['Close'].rolling(window=20).mean()
                ap = mpf.make_addplot(df['MA20'], panel=0, ylabel='MA20', color='green')

                mpf.plot(df, type='candle', volume=True, style='ibd', savefig=graph_filename,
                         addplot=ap)
            else:
                ap = None

            mpf.plot(df, type='candle', volume=True, style='ibd', savefig=graph_filename)
            return render_template('home.html', initial_graph=graph_filename, stock_code=stock_code)
        else:
            return render_template('home.html', error_message=error_message)

    return render_template('home.html')
# 전역 변수 설정
APP_KEY = 'APP_KEY'
APP_SECRET = 'APP_SECRET'
ACCESS_TOKEN = None

# 토큰 발급용 함수
def get_token(APP_KEY, APP_SECRET):
    global ACCESS_TOKEN
    url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"
    payload = json.dumps({
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    })
    headers = {
        'content-type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        data = response.json()
        # json에서 access code 추출
        ACCESS_TOKEN = data['access_token']
    else:
        # 에러 메세지
        print(f"Failed to get token: {response.status_code} {response.text}")




# Vertex AI를 지정된 프로젝트 및 위치로 초기화합니다.
vertexai.init(project="project", location="location")

# gemini-1.5-pro-001 모델을 사용하여 Geget_tokennerativeModel 인스턴스를 생성합니다.
model = GenerativeModel("gemini-1.5-pro-001")

# 콘텐츠 생성을 위한 구성 설정을 정의합니다.
generation_config = {
    "max_output_tokens": 8192, # 필요에 따라 조정합니다.
    "temperature": 1, # 출력의 무작위성 조정
    "top_p": 0.95 # 출력의 다양성 조정
}

# 생성한 콘텐츠의 안전성을 관리하기 위해 안전 설정을 정의합니다.
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# 재무 분석에 대한 컨텍스트를 정의합니다.
prompt_context = (
    "재무 분석가로서 저의 임무는 사용자가 제공하는 재무제표를 분석하고 회사의 재무 건강 상태에 대한 통찰을 제공하는 것입니다. 밑은 예시이고 다른 분야 재무재표도 충분히 분석 할 수 있어야 합니다. "
    "이를 위해 수익성, 유동성 및 자산 품질의 주요 지표 및 트렌드를 강조하겠습니다."
    "**수익성**"
    "**유동성**"
    "**자산 품질**"
    "**주요 트렌드**"
    "**권장 사항**"
    "분석을 바탕으로 명확하고 실행 가능한 추천 사항을 제공합니다. "
    "분석이 상세하면서도 이해하기 쉬워서 사용자가 당신의 통찰을 쉽게 이해하고 적용할 수 있도록 하세요. "
    "사용자가 추가 정보나 설명이 필요할 경우, 이를 제공하여 이해를 돕습니다. "
    "전문적이면서도 친근한 방식으로 사용자가 재무제표를 분석할 수 있도록 도와주세요."
    "반드시 한글로 출력되어야 하고 한국어를 제외한 다른 언어가 입력된다면 결과를 반드시 한글로 출력하세요."
)
#어드민용 DB에 저장
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return '파일이 없습니다.'
    file = request.files['file']
    if file.filename == '':
        return '선택된 파일이 없습니다.'

    filename = os.path.join('uploads', file.filename)
    stock_id = request.form['stock_id']
    date = request.form.get('date', datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs('uploads', exist_ok=True)  # 'uploads' 디렉토리가 없을 경우 생성합니다.
    file.save(filename)
    text = extract_text(filename, codec='utf-8') # 파일에서 텍스트 추출
    print("Extracted Text:\n", text)

    response_text = generate_response_admin(text, stock_id, date)  # 추출된 텍스트를 응답 생성 함수에 전달하여 응답 생성
    data = {
        'message': response_text
    }
    print("Extracted Text:\n", response_text)

    return render_template('result_admin.html', result=response_text)


# 어드민이 사용함
def generate_response_admin(user_input, stock_id, date):
    # 컨텍스트와 사용자 입력을 결합하여 응답을 생성
    full_prompt = f"{prompt_context}\nUser input: {user_input}"

    # 사용자 입력이 비어 있거나 공백으로만 구성되어 있는지 확인합니다.
    if not user_input.strip():
        return "입력은 비워둘 수 없습니다. 유효한 입력을 제공해 주세요."

    # 지정된 구성 및 안전 설정을 사용하여 콘텐츠를 생성합니다.
    responses = model.generate_content(
        [full_prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,  # 단순한 응답을 위해 스트리밍 비활성화
    )

    # 생성된 응답 텍스트에 직접 액세스합니다.
    try:
    # 응답이 단일 GenerationResponse 객체라고 가정하고 .text 속성이 있을 것으로 가정합니다.
        response_text = responses.text
    except AttributeError:
        response_text = "응답이 생성되지 않았거나 응답 형식이 예상과 다릅니다."

    # 데이터베이스 연결 및 커서 생성
    conn = mariadb.connect(**db_config)
    cursor = conn.cursor()

    try:
        # 데이터 삽입 쿼리
        # stockcode =%s 인 곳의 finance_analysis =%s 추가
        insert_query = """
          UPDATE stocklistkor
          SET finance_analysis = %s
          WHERE id = %s;     
          """
        data_to_insert = (response_text, stock_id)

        cursor.execute(insert_query, data_to_insert)
        conn.commit()
        print("stock_list_table 재무재표 삽입 완료")

    except mariadb.Error as e:
        response_text = f"Database operation error: {e}"

    finally:
        # 작업 수행 후 연결 닫기
        cursor.close()
        conn.close()

    conn = mariadb.connect(**db_config)
    cursor = conn.cursor()

    #재무재표 테이블 채우기
    try:
          # 데이터 삽입 쿼리
        insert_query = """
        INSERT INTO analysis (stock_table_id, finance_analysis, date)
        VALUES (%s, %s, %s)
        """
        data_to_insert = (stock_id, response_text, date)

        cursor.execute(insert_query, data_to_insert)
        conn.commit()
        print("Data inserted successfully into 'analysis' table.")

    except mariadb.Error as e:
        response_text = f"Database operation error: {e}"

    finally:
        # 작업 수행 후 연결 닫기
        cursor.close()
        conn.close()

    return response_text

@app.route('/user_finance', methods=['POST'])
def upload_file_user():
    if 'file' not in request.files:
        return '파일이 없습니다.'

    file = request.files['file']

    if file.filename == '':
        return '선택된 파일이 없습니다.'

    filename = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)  # 'uploads' 디렉토리가 없을 경우 생성합니다.
    file.save(filename)
    text = extract_text(filename, codec='utf-8') # 파일에서 텍스트 추출

    print("Extracted Text:\n", text)

    # 콘솔 창에 텍스트 출력
    response_text = generate_response_user(text)  # 추출된 텍스트를 응답 생성 함수에 전달하여 응답 생성
    data = {
        'message': response_text
    }

    print("Extracted Text:\n", response_text)
    return render_template('result.html', result=response_text)

def generate_response_user(user_input):
    # 컨텍스트와 사용자 입력을 결합하여 응답을 생성
    full_prompt = f"{prompt_context}\nUser input: {user_input}"

    # 사용자 입력이 비어 있거나 공백으로만 구성되어 있는지 확인합니다.
    if not user_input.strip():
        return "입력은 비워둘 수 없습니다. 유효한 입력을 제공해 주세요."

    # 지정된 구성 및 안전 설정을 사용하여 콘텐츠를 생성합니다.
    responses = model.generate_content(
        [full_prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,  # 단순한 응답을 위해 스트리밍 비활성화
    )

    # 생성된 응답 텍스트에 직접 액세스합니다.
    try:
        # 응답이 단일 GenerationResponse 객체라고 가정하고 .text 속성이 있을 것으로 가정합니다.
        response_text = responses.text
    except AttributeError:
        response_text = "응답이 생성되지 않았거나 응답 형식이 예상과 다릅니다."

    return response_text



# 금융 분석을 위한 컨텍스트 정의 (한국어)
prompt_context_polio = """


### 총평 ###
**입력 예시**: 하락 예상 또는 상승 예상 또는 중립 
**입력 예시**: 1. 기본 분석, 2. 머신러닝, 3. 뉴스 감정 분석 3가지의 결과를 확인 후 총평을 알려주세요. 

이제 이 프롬프트를 사용하여 하이브리드 전략으로 주식 포트폴리오를 구성해 보세요. 이 전략은 장기적인 성장과 단기적인 기회를 모두 고려하여 최적의 투자 결정을 내릴 수 있도록 도와줍니다. 


"""
@app.route('/uploads', methods=['POST'])
def upload_files():
    stock_id1 = request.form.get('stock_id1')
    stock_id2 = request.form.get('stock_id2')
    stock_id3 = request.form.get('stock_id3')

    stock_ids = [stock_id for stock_id in [stock_id1, stock_id2, stock_id3] if stock_id]

    response_text = generate_response_polio(stock_ids)
    return render_template('result1.html', result=response_text)

def generate_response_polio(stock_ids):
    def get_stock_data(stock_id):
        # 데이터베이스 연결 설정
        db_config = {
            'user': 'user1',
            'password': '1111',
            'host': '192.168.0.110',
            'port': 3307,
            'database': 'pro3'
        }
        conn = mariadb.connect(**db_config)
        cursor = conn.cursor()

        try:
            # stock_id로 stockCode와 finance_analysis 가져오기
            cursor.execute("SELECT stockCode, finance_analysis FROM stocklistkor WHERE id = ?", (stock_id,))
            stock_data = cursor.fetchone()
            print(f"주식 코드 및 금융 분석 쿼리 결과: {stock_data}")  # 콘솔에 출력
            if not stock_data:
                return None, "유효한 stock_id가 아닙니다."

            stock_code, finance_analysis = stock_data
            finance_analysis = finance_analysis if finance_analysis else "금융 분석 데이터가 없습니다."

            # stock_code에 해당하는 분석 데이터 가져오기
            cursor.execute("SELECT finance_analysis FROM analysis WHERE stock_table_id = ?", (stock_id,))
            analysis_data = cursor.fetchone()
            print(f"분석 쿼리 결과: {analysis_data}")  # 콘솔에 출력
            finance_analysis = analysis_data[0] if analysis_data else finance_analysis

            ml = "머신러닝 데이터가 없습니다."
            news = "최근 뉴스가 없습니다."

            if stock_code:
                # 머신러닝 평균값 가져오기
                cursor.execute("""
                    SELECT linear_avg, SVM_avg, RF_avg, LSTM_avg, XGBoost_avg, current_price, date 
                    FROM machine 
                    WHERE stock_code LIKE CONCAT('%', ?, '%')
                    ORDER BY date DESC LIMIT 1""", (stock_code,))
                ml_data = cursor.fetchone()
                print(f"머신러닝 쿼리 결과: {ml_data}")  # 콘솔에 출력
                if ml_data:
                    ml = f"선형 평균: {ml_data[0]}, SVM 평균: {ml_data[1]}, RF 평균: {ml_data[2]}, LSTM 평균: {ml_data[3]}, XGBoost 평균: {ml_data[4]}, 현재 가격: {ml_data[5]}, 날짜: {ml_data[6]}"

                # 뉴스 감성 분석 데이터 가져오기
                cursor.execute("""
                    SELECT positiveRatio, negativeRatio 
                    FROM news 
                    WHERE stockCode LIKE CONCAT('%', ?, '%') 
                    ORDER BY id DESC""", (stock_code,))
                news_data = cursor.fetchall()
                print(f"뉴스 쿼리 결과: {news_data}")  # 콘솔에 출력

                if news_data:
                    news_aggregated = []
                    for news_item in news_data:
                        news_aggregated.append(
                            f"긍정 비율: {news_item[0]}, 부정 비율: {news_item[1]}"
                        )
                    news = "\n".join(news_aggregated)

        except mariadb.Error as e:
            return None, f"오류: {e}"

        finally:
            # 데이터베이스 연결 종료
            cursor.close()
            conn.close()

        return {
            "finance_analysis": finance_analysis,
            "ml": ml,
            "news": news
        }, None

    # 각 주식 ID에 대한 데이터 가져오기
    data_list = []
    for stock_id in stock_ids:
        data, error = get_stock_data(stock_id)
        if error:
            return error
        data_list.append(data)

    # 전체 프롬프트 생성
    prompt_context_polio = "제공된 재무 및 시장 분석 데이터를 바탕으로 응답을 생성하세요."
    full_prompt = f"{prompt_context_polio}\n\n### 사용자 입력:\n"
    for idx, data in enumerate(data_list, 1):
        full_prompt += (
            f"{idx}. 회사 {idx} 재무 제표: {data['finance_analysis']}\n"
            f"   머신러닝: {data['ml']}\n"
            f"   뉴스 감성 분석: {data['news']}\n\n"
        )

    # 응답 생성 시뮬레이션 (실제 모델 또는 콘텐츠 생성 로직으로 대체 필요)
    try:
        # 'model.generate_content' 메서드가 전체 프롬프트를 기반으로 응답을 생성한다고 가정
        responses = model.generate_content(
            [full_prompt],
            generation_config={},  # 실제 생성 구성으로 대체
            safety_settings={},  # 실제 안전 설정으로 대체
            stream=False  # 간단한 응답의 경우 스트리밍 비활성화
        )

        # responses 객체의 구조를 출력하여 확인
        print("응답 객체 유형:", type(responses))
        print("응답 객체 속성:", dir(responses))
        print("응답 객체:", responses)

        # 'responses' 객체에서 텍스트 속성을 직접 접근 (형식에 따라 조정 필요)
        response_text = responses.text  # 또는 responses.text
    except (AttributeError, IndexError) as e:
        response_text = f"응답이 생성되지 않았거나 예상한 형식과 다릅니다. 오류: {e}"

    return response_text

def fetch_stock_data(days):
    # 날짜 지정 days = 변수
    today_now = datetime.datetime.now()
    kskd_s_date = today_now - datetime.timedelta(days=days)
    kskd_e_date = today_now

    # 형식변환
    kskd_start_date = kskd_s_date.strftime('%Y-%m-%d')
    kskd_end_date = kskd_e_date.strftime('%Y-%m-%d')

    # 코스피 코스닥
    kospi_code = '^KS11'
    kosdaq_code = '^KQ11'

    # 주가 정보 받아오기
    kospi_data = yf.download(kospi_code, start=kskd_start_date, end=kskd_end_date)
    kosdaq_data = yf.download(kosdaq_code, start=kskd_start_date, end=kskd_end_date)

    return kospi_data, kosdaq_data, kskd_start_date, kskd_end_date

def create_graph_json(kospi_data, kosdaq_data, kskd_start_date, kskd_end_date):
    if not kospi_data.empty and not kosdaq_data.empty:
        kospi = kospi_data['Adj Close']
        kosdaq = kosdaq_data['Adj Close']

        # 영업일 기준 x축 정규화
        krx = mcal.get_calendar('XKRX')
        schedule = krx.schedule(start_date=kskd_start_date, end_date=kskd_end_date)
        business_days = schedule.index

        # x축 정규화
        kospi = kospi.reindex(business_days).dropna()
        kosdaq = kosdaq.reindex(business_days).dropna()

        # 그래프
        fig = ms.make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.2,
            subplot_titles=("KOSPI", "KOSDAQ")
        )

        fig.add_trace(
            go.Scatter(
                x=kospi.index,
                y=kospi,
                mode='lines',
                name='KOSPI'
            ), row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=kosdaq.index,
                y=kosdaq,
                mode='lines',
                name='KOSDAQ'
            ), row=2, col=1
        )

        fig.update_layout(
            template='plotly_white',
            yaxis1=dict(
                showticklabels=False
            ),
            yaxis2=dict(
                showticklabels=False
            ),
            showlegend=False
        )

        return fig.to_json()
    else:
        return {"error": "No data available for the given date range."}

@app.route('/kospi_kosdaq')
def homes():
    days = 7
    kospi_data, kosdaq_data, start_date, end_date = fetch_stock_data(days)
    graph_json = create_graph_json(kospi_data, kosdaq_data, start_date, end_date)
    return render_template('index_graph.html', graphJSON=graph_json)

@app.route('/update_graph', methods=['POST'])
def update_graph():
    days = int(request.form.get('days'))

    kospi_data, kosdaq_data, start_date, end_date = fetch_stock_data(days)
    graph_json = create_graph_json(kospi_data, kosdaq_data, start_date, end_date)
    return jsonify(graph_json)

@app.route('/prediction_kr', methods=['POST'])
def index():
    # 폼에서 값 전달받기
    today_now = datetime.datetime.now()
    stock_code = request.form['stock_code']
    stock_id = request.form['stock_id']

    # 날짜 문자열을 datetime으로 변환
    S_DATE = today_now - datetime.timedelta(days=1095) # 3년 전부터
    E_DATE = today_now  # 오늘까지
    # 날짜 데이터 변환
    START_DATE = S_DATE.strftime(format='%Y-%m-%d')
    END_DATE = E_DATE.strftime(format='%Y-%m-%d')
    # 파라미터 전달 값
    STOCK_CODE = stock_code
    PARAM = stock_code.split('.')
    PARAM_CODE = PARAM[0]
    STOCK_ID = stock_id
    print(PARAM_CODE)
    # print(STOCK_ID)
    # print(START_DATE)
    # print(END_DATE)
    # print(STOCK_CODE)

    # yfinance 설정
    yf.pdr_override()

    # 주가 데이터 다운로드
    df = yf.download(STOCK_CODE, start=START_DATE, end=END_DATE)
    # print(df)

    # Close 열만 사용하여 데이터프레임 생성
    data = df[['Close']]

    # 데이터 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 학습용 데이터 준비
    X = []
    y = []

    # 60일치의 데이터를 사용해 다음날의 주가를 예측하도록 설정
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i - 60:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # 기본 회귀 모델 생성
    models = {
        'Linear Regression': LinearRegression(),
        'Support Vector Machine': SVR(kernel='rbf')
    }

    # LSTM 모델 설정
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    X_lstm = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 모델 학습 및 예측 결과 저장
    all_predictions = []

    for name, model in models.items():
        for i in range(100):  # 100번 반복 학습 및 예측
            # 새로운 학습 데이터 생성
            random_seed = random.randint(0, 1000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=random_seed)

            # 모델 학습
            model.fit(X_train, y_train)

            # 마지막 60일 데이터로 예측
            last_60_days = scaled_data[-60:].reshape(1, -1)
            next_day_pred_scaled = model.predict(last_60_days)
            next_day_pred = scaler.inverse_transform(next_day_pred_scaled.reshape(-1, 1))

            # 예측 결과 저장 및 출력
            all_predictions.append([name, i + 1, random_seed, next_day_pred[0][0]])
            print(f"{name}, Iteration {i + 1}/ random_state({random_seed}): {next_day_pred[0][0]}")

    # Random Forest 모델 학습 및 예측
    for i in range(100):
        # 다른 랜덤 시드 사용
        random_seed = random.randint(0, 1000)
        # 데이터 다시 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=random_seed)

        # Random Forest 모델 생성 및 학습
        rf_model = RandomForestRegressor(n_estimators=100, random_state=random_seed)
        rf_model.fit(X_train, y_train)

        # 마지막 60일 데이터로 예측
        last_60_days_rf = scaled_data[-60:].reshape(1, -1)
        next_day_pred_scaled_rf = rf_model.predict(last_60_days_rf)
        next_day_pred_rf = scaler.inverse_transform(next_day_pred_scaled_rf.reshape(-1, 1))

        # 예측 결과 저장 및 출력
        all_predictions.append(['Random Forest', i + 1, random_seed, next_day_pred_rf[0][0]])
        print(f"Random Forest, Iteration {i + 1}/ random_state({random_seed}): {next_day_pred_rf[0][0]}")

    # LSTM 모델 학습 및 예측
    for i in range(100):
        lstm_model.fit(X_lstm, y, epochs=2, batch_size=32, verbose=0)
        last_60_days_lstm = scaled_data[-60:].reshape((1, 60, 1))
        next_day_pred_scaled_lstm = lstm_model.predict(last_60_days_lstm)
        next_day_pred_lstm = scaler.inverse_transform(next_day_pred_scaled_lstm.reshape(-1, 1))

        # 예측 결과 저장 및 출력
        all_predictions.append(['LSTM', i + 1, '-', next_day_pred_lstm[0][0]])
        print(f"LSTM, Iteration {i + 1}: {next_day_pred_lstm[0][0]}")

    # XGBoost 모델 학습 및 예측
    for i in range(100):
        # 다른 랜덤 시드 사용
        random_seed = random.randint(0, 1000)
        # 데이터 다시 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=random_seed)

        # XGBoost 모델 생성 및 학습
        xgb_model = XGBRegressor(n_estimators=100, random_state=random_seed)
        xgb_model.fit(X_train, y_train)

        # 마지막 60일 데이터로 예측
        last_60_days_xgb = scaled_data[-60:].reshape(1, -1)
        next_day_pred_scaled_xgb = xgb_model.predict(last_60_days_xgb)
        next_day_pred_xgb = scaler.inverse_transform(next_day_pred_scaled_xgb.reshape(-1, 1))

        # 예측 결과 저장 및 출력
        all_predictions.append(['XGBoost', i + 1, random_seed, next_day_pred_xgb[0][0]])
        print(f"XGBoost, Iteration {i + 1}/ random_state({random_seed}): {next_day_pred_xgb[0][0]}")

    # 예측 결과를 저장할 데이터프레임 생성
    all_predictions_df = pd.DataFrame(all_predictions, columns=['Model', 'Iteration','random_state', 'Prediction'])


    # 예측값의 평균 계산
    linear_avg = np.mean(all_predictions_df.loc[all_predictions_df['Model'] == 'Linear Regression', 'Prediction'])
    SVM_avg = np.mean(all_predictions_df.loc[all_predictions_df['Model'] == 'Support Vector Machine', 'Prediction'])
    RF_avg = np.mean(all_predictions_df.loc[all_predictions_df['Model'] == 'Random Forest', 'Prediction'])
    LSTM_avg = np.mean(all_predictions_df.loc[all_predictions_df['Model'] == 'LSTM', 'Prediction'])
    XGBoost_avg = np.mean(all_predictions_df.loc[all_predictions_df['Model'] == 'XGBoost', 'Prediction'])
    average_prediction = np.mean(all_predictions_df['Prediction'])

    # 예측 결과를 CSV 파일로 저장
    # all_predictions_df.to_csv('all_predictions.csv', index=False)

    # 현재 주가 가져오기
    current_price = data['Close'].iloc[-1]
    user_id = request.form['user_id']
    today = today_now.strftime('%Y%m%d%H%M%S')
    # 데이터베이스 연결 및 커서 생성
    conn = mariadb.connect(**db_config)
    cursor = conn.cursor()

    try:
        # 데이터 삽입 쿼리
        insert_query = """
                  INSERT INTO machine (user_id, stock_code, linear_avg, SVM_avg, RF_avg, LSTM_avg, XGBoost_avg, current_price, date)
                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                  """
        data_to_insert = (user_id, stock_code, linear_avg, SVM_avg, RF_avg, LSTM_avg, XGBoost_avg, current_price, today)

        cursor.execute(insert_query, data_to_insert)
        conn.commit()
        print("Data inserted successfully into 'ML' table.")

    except mariadb.Error as e:
        response_text = f"Database operation error: {e}"

    finally:
        # 작업 수행 후 연결 닫기
        cursor.close()
        conn.close()

    # 예측값과 현재 주가 출력
    print('=================================================예측=================================================')
    print(f'모델A : {linear_avg}')
    print(f'모델B : {SVM_avg}')
    print(f'모델C : {RF_avg}')
    print(f'모델D : {LSTM_avg}')
    print(f'모델E : {XGBoost_avg}')
    print(f"전체 평균 : {average_prediction}")

    next_day_maybe = None
    # 예측값과 현재 주가 비교하여 상승/하락 예측 출력
    if current_price < average_prediction:
        next_day_maybe = '상승 예상'
        print(f"전일 주가 : {current_price}================================================================상승 예상")
    else:
        next_day_maybe = '하락 예상'
        print(f"전일 주가 : {current_price}================================================================하락 예상")

    # 예측값과 현재 주가 출력
    results = f'A={linear_avg}&B={SVM_avg}&C={RF_avg}&D={LSTM_avg}&E={XGBoost_avg}&av={average_prediction}&cu={current_price}&pr={next_day_maybe}'

    # 리디렉션 경로 결정
    return redirect(f'http://localhost:8080/stock/stockview?id={STOCK_ID}&stockcode={PARAM_CODE}&{results}')

def get_code_to_stockcode(user_id):

    try:
        conn = mariadb.connect(**db_config)
        cursor = conn.cursor()

        sql = "SELECT stock_id FROM polio WHERE userid = ?"
        cursor.execute(sql, (user_id,))

        result = cursor.fetchall()

        if result:
            stock_ids = [row[0] for row in result]
            stock_ids_str = ','.join(['?'] * len(stock_ids))
            sql = f"SELECT id, stockcode, listing FROM stocklistkor WHERE id IN ({stock_ids_str})"
            cursor.execute(sql, stock_ids)
            codes = cursor.fetchall()
            combined = [{"stock_id": code[0], "stock_code": f"{code[1]}.{code[2]}"} for code in codes]
            return combined
        else:
            return []

    except mariadb.Error as e:
        return str(e)

    finally:
        if conn:
            conn.close()

def get_yesterday_stock(stock_codes):
    today_now = datetime.datetime.now()
    yesterday = today_now - datetime.timedelta(days=5)
    START_DATE = yesterday.strftime('%Y-%m-%d')
    END_DATE = today_now.strftime("%Y-%m-%d")

    yesterday_prices = []

    for stock_code in stock_codes:
        try:
            df = yf.download(stock_code["stock_code"], start=START_DATE, end=END_DATE)
            if not df.empty:
                yesterday_close = df['Adj Close'].iloc[-1]
                yesterday_prices.append({"stock_code": stock_code["stock_code"], "price": yesterday_close,
                                         "stock_id": stock_code["stock_id"]})
            else:
                print(f"{stock_code['stock_code']}: No price data found for the given date")
                yesterday_prices.append({"stock_code": stock_code["stock_code"], "price": "API Error",
                                         "stock_id": stock_code["stock_id"]})
        except Exception as e:
            print(f"Failed to download data for {stock_code['stock_code']}: {e}")
            yesterday_prices.append(
                {"stock_code": stock_code["stock_code"], "price": None, "stock_id": stock_code["stock_id"]})

    return yesterday_prices

@app.route('/get_port_price', methods=['POST'])
def get_port_price():
        data = request.json
        user_id = data.get('userid')
        if not user_id:
            return jsonify({"error": "userid is required"}), 400

        stock_codes = get_code_to_stockcode(user_id)
        stock_data = get_yesterday_stock(stock_codes)

        # 직렬화 가능한 형식으로 변환
        serialized_data = [{"stock_code": data["stock_code"], "price": data["price"], "stock_id": data["stock_id"]} for
                           data in stock_data]
        print(serialized_data)
        return jsonify(serialized_data)
# 일별 주가 정보 함수
def get_stock_price_period(APP_KEY, APP_SECRET, ACCESS_TOKEN, STOCK_CODE, START_DATE, END_DATE):
    period_stock_data = []

    url = ( f"https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice?fid_cond_mrkt_div_code=J&fid_input_iscd={STOCK_CODE}&fid_input_date_1={START_DATE}&fid_input_date_2={END_DATE}&fid_period_div_code=D&fid_org_adj_prc=1")

    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {ACCESS_TOKEN}',
        'appkey': APP_KEY,
        'appsecret': APP_SECRET,
        'tr_id': 'FHKST03010100'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        print(data['output2'])
        if 'output2' in data:
            stock_df_data = data['output2']
            stock_df_rows = []

            for item in stock_df_data:
                Date = item.get('stck_bsop_date')
                Close = item.get('stck_clpr')
                Open = item.get('stck_oprc')
                High = item.get('stck_hgpr')
                Low = item.get('stck_lwpr')
                Volume = item.get('acml_vol')

                row = [Date, Close, Open, High, Low, Volume]
                stock_df_rows.append(row)

            stock_df = pd.DataFrame(stock_df_rows, columns=['Date', 'Close', 'Open', 'High', 'Low', 'Volume'])
            # print(df['Date'])
            stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%Y%m%d')
            stock_df[['Close', 'Open', 'High', 'Low', 'Volume']] = stock_df[['Close', 'Open', 'High', 'Low', 'Volume']].apply(pd.to_numeric)

            period_stock_data.append(stock_df)
            return period_stock_data
        else:
            print("Unexpected JSON structure:", data)
            return []

    except requests.exceptions.RequestException as e:
        print("Failed to retrieve data:", e)
        return []

# 페이지 로드시 get / 그래프 생성시 post 방식
@app.route('/stock_graph', methods=['GET', 'POST'])
def plot_stock_chart():
    if request.method == 'GET':
        # GET 요청 처리: 입력 폼 보여주기
        stock_code = request.args.get('stockcode')
        return render_template('stock_graph.html', graph_data=None, stock_code=stock_code)
    elif request.method == 'POST':
        # POST 요청 처리: 그래프 생성
        STOCK_START_DATE = request.form['start_date'].replace('-', '')
        STOCK_END_DATE = request.form['end_date'].replace('-', '')

        GRAPH_STOCK_CODE = request.form['stock_code']
        # 입력 값 확인
        # print(f"START_DATE: {START_DATE}, END_DATE: {END_DATE}, STOCK_CODE: {STOCK_CODE}")

        # 함수 실행 토큰 발급 - 데이터 추출
        if ACCESS_TOKEN is None:
            get_token(APP_KEY, APP_SECRET)

        stock_data = get_stock_price_period(APP_KEY, APP_SECRET, ACCESS_TOKEN, GRAPH_STOCK_CODE, STOCK_START_DATE, STOCK_END_DATE)

        # 그래프 생성
        if stock_data:
            graph_df = stock_data[0]
            # print(df)
            # 주식시장 영업일 추출 (x축 정규화)
            krx = mcal.get_calendar('XKRX')
            schedule = krx.schedule(start_date=graph_df['Date'].min(), end_date=graph_df['Date'].max())
            business_days = schedule.index

            # 두 개의 그래프 생성
            fig = ms.make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.7, 0.3]
            )

            # 캔들스틱 차트 ohlc
            fig.add_trace(
                go.Candlestick(
                    x=graph_df['Date'],
                    open=graph_df['Open'],
                    high=graph_df['High'],
                    low=graph_df['Low'],
                    close=graph_df['Close'],
                    increasing=dict(line=dict(color='red', width=2), fillcolor='rgba(255, 0, 0, 1)'),
                    decreasing=dict(line=dict(color='blue', width=2), fillcolor='rgba(0, 0, 255, 1)')
                ), row=1, col=1
            )

            # 막대그래프 판매량
            fig.add_trace(
                go.Bar(
                    x=graph_df['Date'],
                    y=graph_df['Volume']
                ), row=2, col=1
            )

            # 그래프 사이즈 및 레이아웃 설정
            fig.update_layout(
                height=400,
                xaxis1=dict(
                    rangebreaks=[
                        dict(values=pd.date_range(start=graph_df['Date'].min(), end=graph_df['Date'].max()).difference(
                            business_days))
                    ],
                    rangeslider=dict(visible=False)  # 슬라이더 비활성화
                ),
                xaxis2=dict(
                    rangebreaks=[
                        dict(values=pd.date_range(start=graph_df['Date'].min(), end=graph_df['Date'].max()).difference(
                            business_days))
                    ]
                ),
                yaxis1=dict(
                    range=[graph_df['Close'].min() - 2000, graph_df['Close'].max() + 2000]
                ),
                yaxis2=dict(
                    range=[graph_df['Volume'].min() - 50000, graph_df['Volume'].max() + 50000]
                )
            )

            # 그래프를 JSON으로 변환하여 HTML 템플릿에 전달
            graph_data = fig.to_json()
            return render_template('stock_graph.html', graph_data=graph_data)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')