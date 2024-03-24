# -*- coding: utf-8 -*-
import requests
import json

# Flask 웹앱의 URL
url = 'http://127.0.0.1:5000/predict'
#url = 'https://petitionai.azurewebsites.net/predict'
# POST 요청으로 보낼 데이터
user_Id = 672
RecNum = 5
data = {'userId': user_Id, 'RecNum': RecNum}

# JSON 문자열로 직렬화
json_data = json.dumps(data)

# POST 요청 보내기
response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})

# POST 요청 결과 출력하기
print(f"response: {response.json()}")

