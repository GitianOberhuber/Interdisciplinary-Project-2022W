import numpy as np
import pickle
import os
import argparse
import sklearn
from sklearn.linear_model import Ridge
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import diptest
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde

parser = argparse.ArgumentParser(description='predicting_historical_valence')
parser.add_argument('--k', type=str, default=30, help='Neighborhood size')
args = parser.parse_args()

NEIGHBORHOOD_SIZE = int(args.k)
REGRESSION_MODEL_FILENAME = "regression_model.pkl"
BASEPATH_EMBEDDINGS = "resources/word_embeddings/"
FULLDATE_FILENAME = "histWord_fullAvail_1800.pickle"
HISTORICAL_GOLDLABELS_PATH = "resources/historical_gold_lexicon/goldEN.vad"

#generated with ChatGPT
def calculate_derivative(data, dx=1):
    derivative = []
    n = len(data)
    for i in range(1, n - 1):
        derivative_value = (data[i+1] - data[i-1]) / (2*dx)
        derivative.append(derivative_value)
    return derivative


with open(os.path.join(REGRESSION_MODEL_FILENAME), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASEPATH_EMBEDDINGS, FULLDATE_FILENAME), 'rb') as f:
    data = pickle.load(f)

for year in data.keys():
    embeds = data[year].iloc[:, 1:]
    preds = model.predict(embeds)
    data[year]['pred'] = preds

print("Calculating closes neighbors for each word in the embedding-space of each decade...")
closestNeighborsIdxs = {}
for year in data.keys():
    distances_between_vectors = pdist(data[year].iloc[:, 1:], metric="cosine")
    square_distance_matrix = squareform(distances_between_vectors)
    x = np.argsort(square_distance_matrix, axis=1)[:,:NEIGHBORHOOD_SIZE].copy()
    closestNeighborsIdxs[year] = x
    distances_between_vectors, square_distance_matrix = [],[]

print("Calculating Hartigan's D for closes neighborhoods in each decade, approximating derivative of that and finding words with highest summed derivaives ")
res = {}

for inspected_word in data["2012"]['word']:
    res[inspected_word] = [None, {}]
    ps = []
    for inspected_year in list(data.keys()):
        inspected_word_idx = list(data[inspected_year]['word']).index(inspected_word)
        inspected_word_neighbord_idxs = closestNeighborsIdxs[inspected_year][inspected_word_idx]
        x = data[inspected_year].iloc[list(inspected_word_neighbord_idxs), 301]
        dip, pval = diptest.diptest(x)
        res[inspected_word][1][inspected_year] = (np.array(x), np.array(inspected_word_neighbord_idxs), pval)
        ps.append(dip)

    derivative = np.abs(calculate_derivative(ps))
    area_under_curve = sum(derivative)
    res[inspected_word][0] = area_under_curve

sorted_dict = dict(sorted(res.items(), key=lambda x: x[1][0], reverse=True))
top_words = list(sorted_dict.keys())

#indices and predicted valence of words are required for result object, word-embeddings can be discared
data_min = {}
for year in data.keys():
    data_min[year] = data[year].iloc[:, [0,301]]

#Save result object for later exploration
with open("results_k_{}.pkl".format(NEIGHBORHOOD_SIZE), 'wb') as f:
    pickle.dump([closestNeighborsIdxs, top_words, data_min], f)