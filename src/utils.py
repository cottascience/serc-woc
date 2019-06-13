import random

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]

def data_splits(mask_ids, k=10):
    splits = []
    random.shuffle(mask_ids)
    partitions = split_list(mask_ids, wanted_parts=k)
    for i in range(len(partitions)):
        split = {}
        split['test'] = partitions[i]
        if i + 1 < len(partitions):
            split['validation'] = partitions[i+1]
            val_i = i + 1
        else:
            split['validation'] = partitions[0]
            val_i = 0
        train_j = [j for j in range(len(partitions)) if j!= i and j!=val_i]
        split['train'] = [example for j in train_j for example in partitions[j]]
        splits.append(split)
    return splits
