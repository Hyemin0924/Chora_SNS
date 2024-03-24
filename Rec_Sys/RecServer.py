from flask import Flask, request, jsonify
import io
import os
from PostLens import postLens
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from Evaluator import Evaluator
import requests

app = Flask(__name__)

def LoadpostLensData():
    ml = postLens()
    data = ml.loadpostLensLatestSmall()
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadpostLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# SVD++
SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")

# SampleTopNRecs 함수를 변경: 모델 학습 및 추천 호출
evaluator.TrainModels()

@app.route('/predict', methods=['POST'])
def predict():
    # POST 방식으로 전송된 데이터를 추출합니다.
    data = request.json
    userId = data['userId']
    RecNum = data['RecNum']
    
    response = evaluator.GenerateRecommendations(ml, userId, RecNum)
    
    # 분류 결과를 JSON 형식으로 반환합니다.
    return jsonify(response)

if __name__ == '__main__':
    print('Server Run')
    app.run(host='0.0.0.0',port='5000')
    
