import sys
import math
import numpy as np
import pandas as pd

from .layer import Layer
from .activation import fn

class NeuralNetwork:
    def __init__(self):
        # Build layers
        self.layers = []

        # Build weights matrix
        self.weights = []

        # Build activation function array
        self.activation = []

    def add(self,layer):
        # Add new layer
        self.layers.append(np.ones(layer.n_nodes))
        try :
            # Add new activatioin function
            self.activation.append(layer.activation)
            # Add new weight matrix
            self.weights.append(np.zeros((self.layers[-2].size,self.layers[-1].size)))
        except AttributeError:
            pass
        

    def init_weight(self):
        for i in range(len(self.weights)):
            self.weights[i][...] = np.random.random((self.layers[i].size,self.layers[i+1].size)) * math.sqrt(2.0/self.layers[i+1].size)
    
    def get_class_array(self, x):
        return [1 if i == x else 0 for i in range(self.layers[-1].size)]

    def propagate_foward(self, data):
        # Set input layer
        self.layers[0][...] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.layers)):
            # Propagate activity
            self.layers[i][...] = fn[self.activation[i-1]](np.dot(self.layers[i-1],self.weights[i-1]))

        # Return output
        return self.layers[-1]

    def propagate_backward(self, target, learning_rate):
        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*fn[self.activation[-1]](self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.layers)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*fn['d'+self.activation[i-1]](self.layers[i])
            deltas.insert(0,delta)
            
        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += learning_rate*dw

        # Return error
        return (error**2).sum()

    def learn(self, train, test, output_class, kf, **kwargs):
        epochs = kwargs['epochs']
        learning_rate = kwargs['learning_rate']
        self.init_weight()
        hit = 0
        # Train
        
        error_id = []
        error_data = []
        
        for e in range(epochs):
            error = []
            for t in range(len(train)):
                self.propagate_foward(train.iloc[t,:-1])
                err = self.propagate_backward(self.get_class_array(train.iloc[t,-1]), learning_rate)

                percent = t/len(train)
                hashes = '#' * int(round(percent * 50))
                spaces = ' ' * (50 - len(hashes))
                sys.stdout.write('\rEpoch #{}: [{}] {}% | error : {:f}'.format(e,hashes + spaces, int(round(percent*100)), err ))
                sys.stdout.flush()
                error.append(err)
            error = np.average(error)
            print('\rEpoch #{}: [{}] {}% | error : {:f}'.format(e,'#'*50,100,error))

            error_id.append(e)
            error_data.append(error)

        error_id = np.array(error_id)
        error_data = np.array(error_data)

        new = pd.DataFrame({'err_id': error_id, 'error': error_data})
        new.to_csv('error_fold' + str(kf) + '.csv')

        # Test
        for t in range(len(test)):
            o = self.propagate_foward(test.iloc[t,:-1])
            y_nn = list(o).index(np.max(o))
            if y_nn == test.iloc[t,-1]:
                hit+=1
            print('{:>2} :=: {} -> {} expected {}'.format(t,list(test.iloc[t,:-1]),output_class[y_nn],output_class[test.iloc[t,-1]]))
        
        print('Accuracy : {}%'.format(hit/len(test)*100))
        return hit/len(test)*100