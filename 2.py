import os
import json
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import pandas as pd
import re
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, request, render_template, jsonify
import nltk
import mysql.connector

app = Flask(__name__)

# nltk 데이터 다운로드
nltk.download('vader_lexicon')

base_folder = r'D:\11111pro\data'
index_folder = os.path.join(base_folder, 'index')
news_folder = os.path.join(base_folder, 'news')

try:
    os.makedirs(index_folder, exist_ok=True)
    os.makedirs(news_folder, exist_ok=True)
    os.makedirs(os.path.join(index_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(news_folder, 'images'), exist_ok=True)

    # 권한 설정 시도
    os.chmod(base_folder, 0o777)
    os.chmod(index_folder, 0o777)
    os.chmod(news_folder, 0o777)

    print("권한 설정 완료")
except OSError as e:
    print(f"Error: {e}")

# 크롤링
def get_news_title_and_content_and_images(url, img_counter, save_folder):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
    }

    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'lxml')

            # 변수 초기화
            title = ''
            content = ''
            images = []
            image_names = []
            span_text = ''
            date_text = ''

            # 제목 가져오기
            title_element = soup.find('h1', id='articleTitle')
            if title_element:
                title = title_element.text.strip()

            # 내용 가져오기
            content_element = soup.find('div', id='article')
            if content_element:
                content = content_element.text.strip()

            # 이미지 가져오기
            for img in soup.find_all('img'):
                img_url = img.get('src')
                if img_url:
                    full_img_url = urljoin(url, img_url)
                    images.append(full_img_url)
                    try:
                        img_response = requests.get(full_img_url)
                        img_data = img_response.content
                        img_extension = os.path.splitext(img_url)[1]
                        img_name = f'image{img_counter:04d}{img_extension}'
                        img_path = os.path.join(save_folder, 'images', img_name)
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        with open(img_path, 'wb') as f:
                            f.write(img_data)
                        image_names.append(img_name)
                        img_counter += 1
                    except Exception as e:
                        print(f"Failed to download image {full_img_url}: {e}")

            # 기타 정보 가져오기 (예: span_text, date_text)
            span_element = soup.find('span', class_='w-[57px] overflow-hidden overflow-ellipsis whitespace-nowrap text-xs font-semibold')
            span_text = span_element.text.strip() if span_element else ''

            div_elements = soup.find_all('div', class_='flex flex-row items-center')
            for div_element in div_elements:
                span_element = div_element.find('span')
                if span_element:
                    date_text = span_element.text.strip()
                    break

            return title, content, images, image_names, span_text, date_text, img_counter

        else:
            print(f"Failed to fetch page: {res.status_code}")
            return '', '', [], [], '', '', img_counter

    except requests.RequestException as e:
        print(f"Error fetching URL: {url}")
        print(e)
        return '', '', [], [], '', '', img_counter


def save_news_datas(numbers, save_folder, is_index=False):
    img_counter = 1
    all_news_data = []

    for number in numbers:
        url = f'https://kr.investing.com/news/stock-market-news/article-{number}'
        try:
            title, content, images, image_names, span_text, date_text, img_counter = get_news_title_and_content_and_images(
                url, img_counter, save_folder)

            if span_text and title and content and images:
                news_data = {
                    'title': title,
                    'content': content,
                    'image_url': images,
                    'image_file_names': image_names,
                    '종목코드': span_text
                }
                if is_index:
                    news_data['날짜'] = date_text
                all_news_data.append(news_data)
            else:
                print(f"URL {url} 에서 데이터를 가져올 수 없습니다. 정보가 불완전합니다.")
        except Exception as e:
            print(f"URL {url} 에서 데이터를 가져오는 중 에러 발생: {str(e)}")

    save_news_data(all_news_data, save_format='json', save_folder=save_folder)
    save_news_data(all_news_data, save_format='excel', save_folder=save_folder)

def save_news_data(news_data, save_format, save_folder):
    try:
        if save_format == 'json':
            json_path = os.path.join(save_folder, 'news_data.json')
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(news_data, json_file, ensure_ascii=False, indent=4)
            print(f"뉴스 데이터가 '{json_path}'에 JSON 형식으로 저장되었습니다.")
        elif save_format == 'excel':
            df = pd.DataFrame(news_data)
            excel_path = os.path.join(save_folder, 'news_data.xlsx')
            df.to_excel(excel_path, index=False)
            print(f"뉴스 데이터가 '{excel_path}'에 Excel 형식으로 저장되었습니다.")
        else:
            print(f"알 수 없는 저장 형식: {save_format}")
    except Exception as e:
        print(f"데이터 저장 중 에러 발생: {str(e)}")

# 이미지 다운로드
def save_news_data(news_data, save_format, save_folder):
    if save_format == 'json':
        file_path = os.path.join(save_folder, "index_data.json" if 'index' in save_folder else "news_data.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(news_data, f, ensure_ascii=False, indent=4)
    elif save_format == 'excel':
        file_name = "index_data.xlsx" if 'index' in save_folder else "news_data.xlsx"
        file_path = os.path.join(save_folder, file_name)
        df = pd.DataFrame(news_data)
        df['image_file_names'] = df['image_file_names'].apply(lambda x: ', '.join(x))
        df.to_excel(file_path, index=False)

    print(f"뉴스 데이터가 '{file_path}'에 저장되었습니다.")


# 특정 번호만 가져오기
specific_numbers_index = [1140892, 1140625, 1141205, 1139504, 1141955]
specific_numbers_news = [1140597, 1140857, 1139018, 1139150, 1139193, 1139617, 1140550]

# 인덱스 데이터 크롤링 및 저장
print("Index 데이터 크롤링 및 저장 중...")
save_news_datas(specific_numbers_index, index_folder, is_index=True)
print("Index 데이터 저장 완료!")

# 뉴스 데이터 크롤링 및 저장
print("News 데이터 크롤링 및 저장 중...")
save_news_datas(specific_numbers_news, news_folder, is_index=False)
print("News 데이터 저장 완료!")



# 종목명 업데이트
def update_span_text(excel_path, output_excel_path):
    # 파일 읽기
    df = pd.read_excel(excel_path)
    b_df = pd.read_excel("data_0203_20240618.xlsx")
    c_df = pd.read_csv("nasdaq_screener_1718687064986.csv")
    d_df = pd.read_csv("nasdaq_screener_1718687077580.csv")
    e_df = pd.read_csv("nasdaq_screener_1718687077580.csv")

    # 새로운 열 추가
    df['한글 종목명'] = ''
    df['영문 종목명'] = ''

    for index, row in df.iterrows():
        if pd.notna(row['종목코드']):
            span_texts = row['종목코드']

            korean_names = []
            english_names = []

            # '종목코드'가 문자열인 경우 리스트로 변환
            if isinstance(span_texts, str):
                span_texts = [span_texts]
            elif not isinstance(span_texts, list):
                span_texts = [str(span_texts)]

            # 각 종목코드에 대해 처리
            for text in span_texts:
                if '/' in text:
                    df.drop(index, inplace=True)
                    break

                if text == 'KQ11':
                    korean_names.append('코스닥')
                    english_names.append('KOSDAQ')
                elif text == 'KS11':
                    korean_names.append('코스피')
                    english_names.append('KOSPI')
                elif text == 'N11' or text == 'NDX':
                    korean_names.append('나스닥')
                    english_names.append('NDX')
                elif text == 'US500':
                    korean_names.append('S&P500')
                    english_names.append('S&P500')
                elif text.isdigit():
                    num_str = str(text).zfill(6)
                    matching_row = b_df[b_df['단축코드'] == num_str]
                    if not matching_row.empty:
                        korean_name = matching_row['한글 종목명'].iloc[0]
                        english_name = matching_row['영문 종목명'].iloc[0]
                        korean_name = re.sub(r'\b(보통주|Common|Class A|Stock|Ordinary)\b', '', korean_name).strip()
                        english_name = re.sub(r'\b(Common|Class A|Stock|Ordinary)\b', '', english_name).strip()
                        korean_names.append(korean_name)
                        english_names.append(english_name)
                else:
                    matching_row = c_df[c_df['Symbol'] == text]
                    if not matching_row.empty:
                        korean_name = matching_row['Name'].iloc[0]
                        korean_name = re.sub(r'\b(보통주|Common|Class A|Stock|Ordinary)\b', '', korean_name).strip()
                        korean_names.append(korean_name)
                        english_name = korean_name
                        english_names.append(english_name)

                    matching_row = d_df[d_df['Symbol'] == text]
                    if not matching_row.empty:
                        english_name = matching_row['Name'].iloc[0]
                        english_name = re.sub(r'\b(Common|Class A|Stock|Ordinary)\b', '', english_name).strip()
                        english_names.append(english_name)

                    matching_row = e_df[e_df['Symbol'] == text]
                    if not matching_row.empty:
                        english_name = matching_row['Name'].iloc[0]
                        english_name = re.sub(r'\b(Common|Class A|Stock|Ordinary)\b', '', english_name).strip()
                        english_names.append(english_name)

            # 결과 업데이트
            if index in df.index:
                df.at[index, '한글 종목명'] = ', '.join(korean_names)
                df.at[index, '영문 종목명'] = ', '.join(english_names)

    # 결과를 엑셀 파일로 저장
    df.to_excel(output_excel_path, index=False)
    print(f"업데이트된 데이터가 '{output_excel_path}'에 저장되었습니다.")
# index와 news 데이터에 대해 update_span_text 함수 적용
update_span_text(os.path.join(index_folder, "index_data.xlsx"), os.path.join(index_folder, "updated_index_data.xlsx"))
update_span_text(os.path.join(news_folder, "news_data.xlsx"), os.path.join(news_folder, "updated_news_data.xlsx"))

import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# 모델과 토크나이저 로드
model_name = "beomi/kcbert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 감정 분석 파이프라인 초기화
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def sentiment_analysis(content):
    # 문장을 분리하여 각각의 감정 분석
    sentences = content.split('.')  # 문장 분리 기준은 필요에 따라 조정 가능
    sentiment_results = [sentiment_analyzer(sentence[:300]) for sentence in sentences if sentence]

    # 각 문장의 감정을 합산하여 전체 감정 판단
    positive_count = sum(1 for result in sentiment_results if result[0]['label'] == 'LABEL_1')
    negative_count = sum(1 for result in sentiment_results if result[0]['label'] == 'LABEL_0')
    total_sentences = len(sentiment_results)

    pos_ratio = positive_count / total_sentences if total_sentences != 0 else 0
    neg_ratio = negative_count / total_sentences if total_sentences != 0 else 0

    pred_sentiment = 'positive' if pos_ratio > neg_ratio else 'negative' if neg_ratio > pos_ratio else 'neutral'

    return total_sentences, positive_count, negative_count, pos_ratio, neg_ratio, pred_sentiment


def add_sentiment_analysis_to_excel(excel_path, output_excel_path):
    df = pd.read_excel(excel_path)

    # 감정 분석 결과를 담을 새로운 열 초기화
    df['문장 총 개수'] = 0
    df['긍정 문장 개수'] = 0
    df['부정 문장 개수'] = 0
    df['긍정 문장 비율'] = 0.0
    df['부정 문장 비율'] = 0.0
    df['예측 감정'] = ''

    for index, row in df.iterrows():
        content = row['content']
        if not isinstance(content, str):
            continue

        total_sentences, pos_count, neg_count, pos_ratio, neg_ratio, pred_sentiment = sentiment_analysis(content)

        df.at[index, '문장 총 개수'] = total_sentences
        df.at[index, '긍정 문장 개수'] = pos_count
        df.at[index, '부정 문장 개수'] = neg_count
        df.at[index, '긍정 문장 비율'] = pos_ratio
        df.at[index, '부정 문장 비율'] = neg_ratio
        df.at[index, '예측 감정'] = pred_sentiment

    # 새 엑셀 파일로 저장 (경로를 사용자 디렉토리로 변경)
    df.to_excel(output_excel_path, index=False)
    print(f"감정 분석이 추가된 데이터가 '{output_excel_path}'에 저장되었습니다.")


# 뉴스 데이터에 감정 분석 추가
news_folder = "D:\\11111pro\\data\\news"
add_sentiment_analysis_to_excel(os.path.join(news_folder, "updated_news_data.xlsx"),
                                os.path.join(news_folder, "final_news_data.xlsx"))


# 페이지 구성
def format_page_data(excel_path, output_excel_path, page_type):
    df = pd.read_excel(excel_path)

    if page_type == 'news':
        page_columns = ['title', 'content', 'image_url', '단어 총 개수', '긍정 개수', '부정 개수', '긍정 비율', '부정 비율', '예측 감정']
    elif page_type == 'newsdetail':
        page_columns = ['title', 'content', 'image_url', '종목코드', '단어 총 개수', '긍정 개수', '부정 개수', '긍정 비율', '부정 비율', '예측 감정',
                        '조회수']
        df['조회수'] = 0
    elif page_type == 'index':
        page_columns = ['title', 'content']
    elif page_type == 'indexdetail':
        page_columns = ['title', 'content', 'image_url', '종목코드', '날짜']

    df = df[page_columns]
    df.to_excel(output_excel_path, index=False)
    print(f"{page_type} 페이지 데이터가 '{output_excel_path}'에 저장되었습니다.")


# 페이지 데이터 구성
format_page_data(os.path.join(news_folder, "formatted_news_data.xlsx"), os.path.join(news_folder, "page_news.xlsx"),
                 'news')
format_page_data(os.path.join(news_folder, "formatted_news_data.xlsx"),
                 os.path.join(news_folder, "page_newsdetail.xlsx"), 'newsdetail')
format_page_data(os.path.join(index_folder, "formatted_index_data.xlsx"), os.path.join(index_folder, "page_index.xlsx"),
                 'index')
format_page_data(os.path.join(index_folder, "formatted_index_data.xlsx"),
                 os.path.join(index_folder, "page_indexdetail.xlsx"), 'indexdetail')

db_config = {
    'user': 'user3',  # 사용자 이름
    'password': '1111',  # 비밀번호
    'host': '192.168.0.110',  # 호스트 (MariaDB가 실행 중인 서버의 IP 주소)
    'port': 3307,  # 포트 번호
    'database': 'pro3',  # 데이터베이스 이름
    'charset': 'utf8mb4'  # 문자셋
}


def insert_index_data_to_db(excel_path, db_config):
    try:
        # 연결 생성
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        excel_path="/data/index/updated_index_data.xlsx"

        # Excel 파일 읽기
        df = pd.read_excel(excel_path)

        # NaN 값 처리 (빈 문자열로 대체)
        df.fillna('', inplace=True)

        # 데이터 삽입
        for index, row in df.iterrows():
            sql = '''
            INSERT INTO indexnews (title, content, images, imageFileNames, stockCode, koreanStockName, 
                                   englishStockName, date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            '''
            values = (row['title'], row['content'], row['image_url'], row['image_file_names'], row['종목코드'],
                      row['한글 종목명'], row['영문 종목명'], row['날짜'])

            cursor.execute(sql, values)

        # 변경 사항 저장 및 연결 종료
        conn.commit()
        print("기존 데이터를 삭제하고 새로운 데이터를 성공적으로 삽입하였습니다.")

    except mysql.connector.Error as e:
        print(f"Error: {e}")

    finally:
        # 연결 종료
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()


# 인덱스 데이터베이스 삽입 실행
index_excel_path = 'D:/11111pro/data/index/updated_index_data.xlsx.xlsx'
insert_index_data_to_db(index_excel_path, db_config)

def insert_news_data_to_db(excel_path, db_config):
    try:
        # 연결 생성
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        excel_path = "D:/11111pro/data/index/updated_news_data.xlsx"
        # Excel 파일 읽기
        df = pd.read_excel(excel_path)

        # NaN 값 처리 (빈 문자열로 대체)
        df.fillna('', inplace=True)

        # 데이터 삽입
        for index, row in df.iterrows():
            sql = '''
            INSERT INTO news (title, content, images, imageFileNames, stockCode, koreanStockName, 
                              englishStockName, wordCount, positiveWords, positiveCount, negativeWords, 
                              negativeCount, positiveRatio, negativeRatio, sentimentPrediction)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            values = (row['title'], row['content'], row['image_url'], row['image_file_names'], row['종목코드'],
                      row['한글 종목명'], row['영문 종목명'], row['단어 총 개수'], row['긍정 단어'], row['긍정 개수'],
                      row['부정 단어'], row['부정개수'], row['긍정 비율'], row['부정 비율'],
                      row['예측 감정'])

            cursor.execute(sql, values)

        # 변경 사항 저장 및 연결 종료
        conn.commit()
        print("기존 데이터를 삭제하고 새로운 데이터를 성공적으로 삽입하였습니다.")

    except mysql.connector.Error as e:
        print(f"Error: {e}")

    finally:
        # 연결 종료
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()


# 뉴스 데이터베이스 삽입 실행
news_excel_path = 'D:/11111pro/data/news/final_news_data.xlsx'
insert_news_data_to_db(news_excel_path, db_config)

def load_index_data(excel_path):
    try:
        df = pd.read_excel(excel_path)
        data = df.to_dict(orient='records')
    except Exception as e:
        print(f"Error reading {excel_path}: {e}")
        data = []
    return data
@app.route('/index')
def index():
    index_data = fetch_data_from_db('indexnews')
    return render_template('index.html', index_data=index_data)

@app.route('/news')
def news():
    news_data = fetch_data_from_db('news')
    return render_template('news.html', news_data=news_data)

@app.route('/newsDetail/<int:id>')
def news_detail(id):
    news_detail = fetch_detail_from_db('news', id)
    return render_template('newsDetail.html', news=news_detail)  # 'news'라는 이름으로 전달

@app.route('/indexDetail/<int:id>')
def index_detail(id):
    index_detail = fetch_detail_from_db('indexnews', id)
    return render_template('indexDetail.html', index_detail=index_detail)

def fetch_data_from_db(table_name):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"SELECT * FROM {table_name}")
        data = cursor.fetchall()
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        data = []
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
    return data

def fetch_detail_from_db(table_name, record_id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"SELECT * FROM {table_name} WHERE id = %s", (record_id,))
        data = cursor.fetchone()  # 단일 레코드를 가져옴
    except mysql.connector.Error as e:
        print(f"Error fetching data from database: {e}")
        data = None  # 오류 발생 시 None을 반환하도록 처리
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
    return data


# 서버 실행
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')