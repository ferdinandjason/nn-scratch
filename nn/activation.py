import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

def relu(x, a=0.01):
    return np.maximum(x, a*x)

def drelu(x, a=0.01):
    return [1.0 if (i>=0) else a for i in x]
fn = {
    'sigmoid':sigmoid,
    'dsigmoid':dsigmoid,
    'relu':relu,
    'drelu':drelu
}
