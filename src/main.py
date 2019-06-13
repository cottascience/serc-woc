import numpy as np
import sys, torch, random, utils, plots, copy
import torch.nn.functional as F
from model import Model
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import scipy.stats
from sys import argv

PATIENCE = 20

def compute_acc(pred, true):
    acc = []
    for c in range(np.size(true,1)):
        acc.append(accuracy_score(true[:,c], pred[:,c]))
    return acc

def compute_rec(pred, true):
    rec = []
    for c in range(np.size(true,1)):
        rec.append(recall_score(true[:,c], pred[:,c]))
    return rec

def main():
    if argv[1] == "JOINTP" or argv[1] == "RR":
        from parser_joint import Data
    else:
        from parser import Data
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    data = Data("../data/", device, argv[1])

    ids = range(data.num_examples)
    splits = utils.data_splits(ids,k=5)
    acc = []
    rec = []
    RR = False
    if argv[1][:2] == "RR":
        RR = True

    for split in splits:
        data.reset()
        data.generate_dataset(split["train"])
        data.generate_dataset(split["validation"], split="val")
        data.generate_dataset(split["test"], split="test")

        if len(data.y.size()) == 1:
            data.y = data.y.view(len(data.y),1)
            data.prev_y = data.prev_y.view(len(data.prev_y),1)
            data.val_y = data.val_y.view(len(data.val_y),1)
            data.val_prev_y = data.val_prev_y.view(len(data.val_prev_y),1)
            data.test_y = data.test_y.view(len(data.test_y),1)
            data.test_prev_y = data.test_prev_y.view(len(data.test_prev_y),1)

        num_exs = float(data.y.size(0) + data.test_y.size(0) + data.val_y.size(0))
        class_balance = np.sum(data.y.cpu().detach().numpy(), axis=0) + np.sum(data.test_y.cpu().detach().numpy(), axis=0) + np.sum(data.val_y.cpu().detach().numpy(), axis=0)
        class_balance = class_balance/num_exs

        print "Class Balance:\t" + str(class_balance)

        model = Model(input_size=data.num_features, hidden_size=16, output_size=data.y.size(1) , units=data.x_sizes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best_val = 0
        best_step = 0
        train_hist = []
        test_hist = []
        for step in range(1, 301):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(data.x, data.x_ids, data.x_sizes, data.prev_y), data.y.float())
            loss.backward()
            optimizer.step()
            "train acc"
            pred = torch.round(torch.sigmoid(model(data.x, data.x_ids, data.x_sizes, data.prev_y)))
            train_acc = compute_acc(pred.cpu().detach().numpy(), data.y.cpu().detach().numpy())
            "val acc"
            pred = torch.round(torch.sigmoid(model(data.val_x, data.val_x_ids, data.x_sizes, data.val_prev_y)))
            val_acc = compute_acc(pred.cpu().detach().numpy(), data.val_y.cpu().detach().numpy())
            "test acc"
            pred = torch.round(torch.sigmoid(model(data.test_x, data.test_x_ids, data.x_sizes, data.test_prev_y)))
            test_acc = compute_acc(pred.cpu().detach().numpy(), data.test_y.cpu().detach().numpy())
            test_rec = compute_rec(pred.cpu().detach().numpy(), data.test_y.cpu().detach().numpy())

            if sum(val_acc) >= best_val:
                best_val = sum(val_acc)
                final_test = test_acc
                final_rec = test_rec
                best_model = copy.deepcopy(model.state_dict())
                best_step = step

            # if step > best_step:
            #     break
        print final_test
        acc.append(final_test)
        rec.append(final_rec)

    print argv[1] + "\t ACC \t MEAN:\t" + str(np.mean(np.asarray(acc), axis=0)) + "\STD:\t" + str(np.std(np.asarray(acc), axis=0))
    print argv[1] + "\t RECALL \t MEAN:\t" + str(np.mean(np.asarray(rec), axis=0)) + "\STD:\t" + str(np.std(np.asarray(rec), axis=0))

if __name__== "__main__":
    main()
