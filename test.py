'''
import pandas as pd

# Excel 파일에서 데이터 불러오기
excel_file_path = 'D:/1teampro/11pro/index0704.xlsx'
df = pd.read_excel(excel_file_path)

# 열 이름을 디버깅하기 위해 출력하기
print("DataFrame의 열 이름:")
print(df.columns)
import requests
from bs4 import BeautifulSoup

url = 'https://kr.investing.com'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
}

try:
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # 웹페이지에서 필요한 데이터 추출
        soup = BeautifulSoup(response.content, 'html.parser')
        # 데이터 처리 로직 추가
        print("크롤링 성공")
    else:
        print(f"서버 응답 오류: {response.status_code}")

except requests.RequestException as e:
    print(f"요청 실패: {e}")
import requests
from bs4 import BeautifulSoup

base_url = 'https://newsis.com/view/?id=NISX20230228_0002208506&cID=13001&pID=13000'

resp = requests.get(base_url) # url에서 데이터 가져오기


print("\n기사 제목 찾기 : ")
soup = BeautifulSoup(resp.text,'lxml')
title_element = soup.find('p',class_='tit title_area')
if title_element:
    title = title_element.text.strip()
    print(title)


print("\n기사 날짜찾기 : ")
date_element = soup.find('p',class_='txt')
if date_element:
    date = date_element.text.strip()
    print(date)

import os
import pandas as pd

def save_index_data(data, excel_path):
    try:
        df = pd.DataFrame(data)
        df.to_excel(excel_path, index=False)
        print(f"Data saved to {excel_path}")
    except Exception as e:
        print(f"Error saving data to {excel_path}: {e}")

# data 폴더 밑에 index 폴더와 news 폴더가 있다고 가정
base_folder = '\data'
index_folder = os.path.join(base_folder, 'index')
news_folder = os.path.join(base_folder, 'news')

# 폴더에 대한 권한 설정 및 폴더 생성
try:
    os.makedirs(index_folder, exist_ok=True)
    os.makedirs(news_folder, exist_ok=True)
    os.chmod(base_folder, 0o777)
    os.chmod(index_folder, 0o777)
    os.chmod(news_folder, 0o777)
except OSError as e:
    print(f"Error: {e}")

# index 폴더 밑에 index_data.xlsx 파일 생성
excel_path = os.path.join(index_folder, 'index_data.xlsx')

# 테스트 데이터 생성
test_data = [
    {'ID': 1, 'Name': 'John'},
    {'ID': 2, 'Name': 'Jane'}
]

# 데이터 저장
save_index_data(test_data, excel_path)

# 데이터 로드 테스트
def load_index_data(excel_path):
    try:
        df = pd.read_excel(excel_path)
        data = df.to_dict(orient='records')
    except Exception as e:
        print(f"Error reading {excel_path}: {e}")
        data = []
    return data

index_data = load_index_data(excel_path)
print(index_data)

import socket

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

print(f"Local IP address: {local_ip}")
'''

