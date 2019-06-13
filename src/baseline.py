import numpy as np
import sys, torch, random, utils, plots, copy
import torch.nn.functional as F
from parser import Data
from model import Model
from collections import defaultdict
from sys import argv
import matplotlib.pyplot as plt; plt.rcdefaults()
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import scipy.stats
from sklearn.linear_model import LogisticRegression

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

data = Data("../data/", "cpu", argv[1])
ids = range(data.num_examples)
splits = utils.data_splits(ids,k=5)

acc = []
rec = []


# conditional independence assumptions => weaker
# set => stronger
#


RR = False
if argv[1][:2] == "RR":
    RR = True

for split in splits:
    data.reset()
    data.generate_dataset(split["train"])
    data.generate_dataset(split["validation"], split="val")
    data.generate_dataset(split["test"], split="test")

    y = data.y.float().cpu().detach().numpy()
    x = data.x.float().cpu().detach().numpy()
    new_x = []
    start = 0
    for group in data.x_ids:
        end = start + group[1]
        new_x.append(np.sum(x[start:end,:],axis=0).tolist())
        start = end
    X = np.asarray(new_x)

    if not RR: X = np.concatenate((X, data.prev_y.reshape(len(data.prev_y),1)), axis=1)

    clf = LogisticRegression(solver='lbfgs',class_weight='balanced').fit(X, y)

    y = data.test_y.float().cpu().detach().numpy()
    x = data.test_x.float().cpu().detach().numpy()
    new_x = []
    start = 0
    for group in data.test_x_ids:
        end = start + group[1]
        new_x.append(np.sum(x[start:end,:],axis=0).tolist())
        start = end
    X = np.asarray(new_x)

    if not RR: X = np.concatenate((X, data.test_prev_y.reshape(len(data.test_prev_y),1)), axis=1)

    acc.append(clf.score(X, y))
    rec.append(recall_score(y,clf.predict(X)))

print argv[1] + " ACC " + str(np.asarray(acc).mean()) + " " + str(np.asarray(acc).std())
print argv[1] + " REC " + str(np.asarray(rec).mean()) + " " + str(np.asarray(rec).std())
