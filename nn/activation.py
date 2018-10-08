import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(x, 0)

def drelu(x):
    return 1.0 * (x > 0)

fn = {
    'sigmoid':sigmoid,
    'dsigmoid':dsigmoid,
    'relu':relu,
    'drelu':drelu
}
