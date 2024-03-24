# -*- coding: utf-8 -*-

from PostLens import postLens
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from Evaluator import Evaluator

def LoadpostLensData():
    ml = postLens()
    print("Loading post ratings...")
    data = ml.loadpostLensLatestSmall()
    print("\nComputing post popularity ranks so we can measure novelty later...")
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

# 추천 수행 (ml, userId, numRecs)
evaluator.GenerateRecommendations(ml, 672, 5)

