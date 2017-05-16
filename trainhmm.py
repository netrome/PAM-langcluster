import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import hmmlearn.hmm as hmm

from utils import get_annotated_features
import sys
import numpy as np
import pickle
from multiprocessing import Pool
import os


# Get data and metadata
data, meta = get_annotated_features(sys.argv[1])
meta = meta[1:] # Drop title row
languages = [a[0] for a in meta]
unique = set(languages)
lang_data = {}
hmms = {}

# Initialize empty list and hmm for each language
for lang in unique:
    lang_data[lang] = []
    hmms[lang] = hmm.GMMHMM(n_components=33, n_mix=11)

# Append data to dictionary
for i, lang in enumerate(languages):
    lang_data[lang].append(data[i])

# Concatenate samples in dict
for lang in unique: 
    lang_data[lang] = np.concatenate(lang_data[lang], axis=0) #Explicit is better than implicit

#method for training hmm
def train(lang):
    global lang_data
    global hmms
    print("Training HMM on ", lang)
    hmms[lang].fit(lang_data[lang])
    return hmms[lang]

with Pool(os.cpu_count()) as p:
    hmm_list = p.map(train,unique) 
    for i, lang in enumerate(unique):
       hmms[lang] = hmm_list[i] 
        

print("gonna save")

data_name = sys.argv[1].split("/")[-1].split(".")[0]

pickle.dump(hmms, open("saved_hmms/" + data_name + ".pickle", "wb"))
# Train and save hmms



