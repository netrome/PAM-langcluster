import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import hmmlearn.hmm as hmm
from utils import get_annotated_features
import sys
import numpy as np
import pickle

# Get data
data, meta = get_annotated_features(sys.argv[1])
meta = meta[1:] # Drop title row
languages = [a[0] for a in meta]
pred_languages = ["nope" for a in meta]
unique = set(languages)

# Get hmms
hmms = pickle.load(open("saved_hmms.pickle", "rb"))

# Evaluate hmms
score = 0
for i, sample in enumerate(data):
    print("-------", languages[i])
    max_val = -np.inf
    for lang in hmms:
        prob = hmms[lang].score(sample)
        print(lang, " score: ", prob)

        if prob > max_val:
            max_val = prob
            pred_languages[i] = lang
    if pred_languages[i] == languages[i]:
        score += 1

score /= len(languages)
print("<<<<<<<<<<<<<<<<<<<  HERE COMES THE SCORE, DUBIDUBI  >>>>>>>>>>>>>>>>>>>>>")
print(score)


    
