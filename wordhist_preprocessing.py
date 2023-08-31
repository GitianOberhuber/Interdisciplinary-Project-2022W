import numpy as np
import pickle
import argparse
import os
import pandas as pd

BASEPATH = "resources/word_embeddings"
EMBED_IDXS = range(1,300)
WORD_EMBEDDINGS_FILENAME_TEMPLATE = os.path.join(BASEPATH, "{}-w.npy")
VOCAB_FILENAME_TEMPLATE = os.path.join(BASEPATH, "{}-vocab.pkl")

parser = argparse.ArgumentParser(description='wordhist_preprocessing')
parser.add_argument('--startYear', type=str, default="1800", help='The year starting from which a subset of words that are available for decades shall be extracted from the XANEW data')
args = parser.parse_args()


hist_embeddings = {}
startYear = args.startYear

print("Creating subset of Histwords containing only words that are available for all decades starting from " + startYear + " ...")

for year in [str(year) for year in range(int(startYear), 1991, 10)]:
    word_embeddings_filename = WORD_EMBEDDINGS_FILENAME_TEMPLATE.format(year)
    vocab_filename = VOCAB_FILENAME_TEMPLATE.format(year)
    df = pd.DataFrame(np.load(word_embeddings_filename))
    df['word'] = pickle.load(open(vocab_filename, "rb"))

    # reorder so "word" is the first column
    cols = df.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    hist_embeddings[year] = df

# build vocab of words that are non-zero over all decades
vocabs_decades = {}
for year in hist_embeddings.keys():
    nonzero_indices_startYear = np.where(np.sum(hist_embeddings[year][EMBED_IDXS], axis=1) != 0)[0]
    nonzero_vocab_startYear = [hist_embeddings[year]["word"][i] for i in nonzero_indices_startYear]
    vocabs_decades[year] = set(nonzero_vocab_startYear)

vocab_fully_available = set.intersection(*vocabs_decades.values())

# filter and keep only words in the combined vocab
for year in hist_embeddings.keys():
    hist_embeddings[year] = hist_embeddings[year].loc[hist_embeddings[year]["word"].isin(vocab_fully_available)]
    hist_embeddings[year] = hist_embeddings[year].sort_values(by="word")

# check if everything worked
dicts = [set(p["word"]) for p in hist_embeddings.values()]
bools = []
for indivDict in dicts:
    bools.append(set(indivDict) == set(dicts[0]))

print("All decades share same dictionary:", all(bools))
print("Size of that shared dictionary:", len(hist_embeddings['1800']))

outpath = os.path.join(BASEPATH, ('histWord_fullAvail_' + str(startYear) + '.pickle'))
with open(outpath, 'wb') as f:
    pickle.dump(hist_embeddings, f)
print("Finished creating subset of Histwords")
print("Saved result to " + outpath)