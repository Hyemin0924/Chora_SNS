# -*- coding: utf-8 -*-
from MovieLens import MovieLens
from surprise import SVD
from surprise import NormalPredictor
from Evaluator import Evaluator
from surprise.model_selection import GridSearchCV
import matplotlib.pyplot as plt

import random
import numpy as np
import pandas as pd

# Parameter
testsubject = 1

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

print("Searching for best parameters...")
param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010],
              'n_factors': [50, 100]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(evaluationData)

# best RMSE score
print("Best RMSE score attained: ", gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

params = gs.best_params['rmse']
SVDtuned = SVD(n_epochs = params['n_epochs'], lr_all = params['lr_all'], n_factors = params['n_factors'])
evaluator.AddAlgorithm(SVDtuned, "SVD - Tuned")

SVDUntuned = SVD()
evaluator.AddAlgorithm(SVDUntuned, "SVD - Untuned")

# 랜덤 추천 추가
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml, testsubject)

# 이전 코드에서 추천 결과를 가져옴
recommendations = evaluator.GetRecommendations(testsubject, k=10)

# Convert recommendations to DataFrame
df_recommendations = pd.DataFrame(recommendations)

# Transpose DataFrame to have algorithms as columns and movie ratings as rows
df_recommendations = df_recommendations.transpose()

# Rename columns for better readability
df_recommendations.columns = [f"Top_{i+1}_Movie_ID" for i in range(10)]
df_recommendations.index.name = 'Algorithm'

# Print DataFrame
print(df_recommendations)