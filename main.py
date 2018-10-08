from nn.neuralnetwork import NeuralNetwork
from nn.dataset import Dataset
from nn.layer import Layer
from nn.utils import kfold_cv_generator

import numpy as np


if __name__ == '__main__':
    dataset = Dataset('iris.csv')
    data = dataset.data
    output_real = dataset.attr

    nn = NeuralNetwork()
    nn.add(Layer(4))
    nn.add(Layer(32,activation='relu'))
    nn.add(Layer(3,activation='sigmoid'))

    id_train, id_test = kfold_cv_generator(data, n_splits=8)
    
    kf = 1
    acc = []
    for train_idx, test_idx in zip(id_train, id_test):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]

        print("#FOLD: ",kf)
        score = nn.learn(train, test, output_real, epochs=100, learning_rate=0.1)
        acc.append(score)

        print()
        kf+=1
    
    print('Accuracy Avg: {}%'.format(np.mean(acc)))