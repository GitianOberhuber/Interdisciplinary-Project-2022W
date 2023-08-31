import numpy as np
import pickle
import pandas as pd
import os
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import csv

BASEPATH_RESOURCES = "resources/"
BASEPATH_EMBEDDINGS = os.path.join(BASEPATH_RESOURCES, "word_embeddings/")
BASEPATH_XANEW = os.path.join(BASEPATH_RESOURCES, "XANEW_lexicon/")
WORD_EMBEDDINGS_FILENAME_TEMPLATE = os.path.join(BASEPATH_EMBEDDINGS, "{}-w.npy")
VOCAB_FILENAME_TEMPLATE = os.path.join(BASEPATH_EMBEDDINGS, "{}-vocab.pkl")
FULLY_AVAILABLE_FILENAME = "histWord_fullAvail_1800.pickle"

print("Reading XANEW lexicon and vocabulary of fully available words from historical word-embeddings...")

xanew_csv_location = os.path.join(BASEPATH_XANEW, 'Ratings_Warriner_et_al.csv')
df_xanew = pd.read_csv(xanew_csv_location, index_col=0)
df_xanew=df_xanew[['Word','V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']]
df_xanew.columns=['word', 'valence', 'arousal', 'dominance']
df_xanew.set_index('word',inplace=True)
df_xanew = df_xanew['valence']
df_xanew = df_xanew[0:]

embeds2012filename = os.path.join(BASEPATH_EMBEDDINGS, "US_Ngrams_2000_12.csv")
vocab_xanew_set = set(df_xanew.index)

with open(os.path.join(BASEPATH_EMBEDDINGS, FULLY_AVAILABLE_FILENAME), 'rb') as f:
    data_fully_available = pickle.load(f)
vocab_fully_available = set(data_fully_available['1990']["word"]) #doesn't actually matter which year as long as it is present

print("Reading contemporary word-embeddings (Geometry of Culture), keeping only words that are present in either XANEW lexicon or historical word-embeddigns ...")

chunks = []
condition = lambda x: x.iloc[0] in vocab_xanew_set or x.iloc[0] in vocab_fully_available
for chunk in pd.read_csv(embeds2012filename, chunksize=10000):
    filtered_chunk = chunk[chunk.apply(condition, axis=1)]
    chunks.append(filtered_chunk)

embeds2012df = pd.concat(chunks, ignore_index=True)
embeds2012df.rename(columns={embeds2012df.columns[0]: 'word'}, inplace=True)

with open(VOCAB_FILENAME_TEMPLATE.format("2012"), 'wb') as f:
    pickle.dump(list(embeds2012df["word"]), f)

with open(WORD_EMBEDDINGS_FILENAME_TEMPLATE.format("2012"), 'wb') as f:
    np.save(f, embeds2012df.iloc[:, 1:])

print("Created subset of contemporary embeddings")

print("Aligning 2012 embeddings to 1990 embeddings...")
# embedding-space-alignment from:
# https://github.com/williamleif/histwords/blob/31e4d200310ebd4051776828eccb8b60c2120427/vecanalysis/seq_procrustes.py

from vecanalysis import alignment
from representations.representation_factory import create_representation
from ioutils import write_pickle, words_above_count, mkdir

embeds2012filename = os.path.join(BASEPATH_EMBEDDINGS, "US_Ngrams_2000_12.csv")


def align_years(years, rep_type, in_dir, out_dir, words, **rep_args):
    first_iter = True
    base_embed = None
    for year in years:
        print("Loading year:", year)
        year_embed = create_representation(rep_type, in_dir + str(year), **rep_args)
        year_words = words[str(year)]
        year_embed.get_subembed(year_words)
        print("Aligning year:", year)
        if first_iter:
            aligned_embed = year_embed
            first_iter = False
        else:
            aligned_embed = alignment.smart_procrustes_align(base_embed, year_embed)
        base_embed = aligned_embed
        print("Writing year:", year)
        foutname = out_dir + str(year)
        np.save(foutname + "-w.npy", aligned_embed.m)
        write_pickle(aligned_embed.iw, foutname + "-vocab.pkl")


align_years([1990, 2012], 'word2vec', BASEPATH_EMBEDDINGS, BASEPATH_EMBEDDINGS + "aligned_",
            {'1990': vocab_fully_available, '2012': list(embeds2012df['word'])})

print("Finished aligning embeddings")

embeds2012df, chunks = [], []

os.remove(WORD_EMBEDDINGS_FILENAME_TEMPLATE.format("2012"))
os.remove(VOCAB_FILENAME_TEMPLATE.format("2012"))

print("Training regression model on XANEW lexicon and aligned 2012 embeddings...")

vocab2012 = pickle.load(open(os.path.join(BASEPATH_EMBEDDINGS, "aligned_2012-vocab.pkl"), "rb"))
embeds2012 = np.load(open(os.path.join(BASEPATH_EMBEDDINGS, "aligned_2012-w.npy"),  "rb"))

embeds2012df = pd.DataFrame(np.column_stack((vocab2012, embeds2012)))
embeds2012df.rename(columns={embeds2012df.columns[0]: 'word'}, inplace=True)
embeds2012df.iloc[:, 1:] = embeds2012df.iloc[:, 1:].astype(float)


data = embeds2012df.merge(df_xanew, left_on=embeds2012df.columns[0], right_index=True, how = "left")
data = data.dropna()

reg = Ridge(alpha = 100)
reg.fit(data.iloc[:,1:301].to_numpy(), data['valence'])

with open("regression_model.pkl", 'wb') as f:
    pickle.dump(reg, f)

print("Finished training regression model, saved to 'regression_model.pkl'")
print("Reducing 2012 embeddings to only those words also present in the historical word-embeddings and adding everything together for later analysis...")

#keeping only words from 2012 embeddings that are also available for the other decades
embeds2012df = embeds2012df[embeds2012df['word'].isin(vocab_fully_available)]
embeds2012df = embeds2012df.sort_values(by = "word")

#keeping only words for other decades that are also available for 2012 embeddings
reduced_2012_wordset = set(embeds2012df["word"])

for year in data_fully_available.keys():
    data_fully_available[year] = data_fully_available[year].loc[data_fully_available[year]["word"].isin(reduced_2012_wordset)]
    data_fully_available[year] = data_fully_available[year].sort_values(by = "word")

data_fully_available['2012'] = embeds2012df

with open(os.path.join(BASEPATH_RESOURCES, "fullAvalList.pkl"), 'wb') as f:
    pickle.dump(reduced_2012_wordset, f)

with open(os.path.join(BASEPATH_EMBEDDINGS, FULLY_AVAILABLE_FILENAME), 'wb') as f:
    pickle.dump(data_fully_available, f)
