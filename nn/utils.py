import numpy as np
import pandas as pd
import random

def kfold_cv_generator(data, n_splits):
    data_split = []
    data_id = [i for i in range(len(data))]
    data_id_copy = data_id.copy()

    train_id = []
    test_id = []

    for i in range(n_splits):
        fold_size = int(len(data)/n_splits)
        if i == n_splits-1:
            fold_size = len(data)-(n_splits-1)*(fold_size)
        
        data_idx = data_id_copy.copy()
        random.shuffle(data_id)
        
        test = data_id[:fold_size]
        for x in test:
            data_idx.remove(x)
            data_id.remove(x)
        
        train = data_idx.copy()

        train_id.append(train)
        test_id.append(test)

    return train_id, test_id


