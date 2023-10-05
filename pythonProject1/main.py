import numpy as np
import math

import pathlib


def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp = np.exp(x - x.max())
    return exp / exp.sum()


dimes = [28 * 28, 10]
activation = [tanh, softmax]
distutils = [
    {'b': [0, 0]},
    {'b': [0, 0], 'w': [-math.sqrt(6 / (dimes[0] + dimes[1])), math.sqrt(6 / (dimes[0] + dimes[1]))]},
]


def init_parameters_b(layer):
    dist = distutils[layer]['b']
    return np.random.rand(dimes[layer]) * (dist[1] - dist[0]) + dist[0]


def init_parameters_w(layer):
    dist = distutils[layer]['w']
    return np.random.rand(dimes[layer - 1], dimes[layer]) * (dist[1] - dist[0]) + dist[0]


def init_parameters():
    parameters = []
    for i in range(len(distutils)):
        layer_parameters = {}
        for j in distutils[i].keys():
            if j == 'b':
                layer_parameters['b'] = init_parameters_b(i)
            elif j == 'w':
                layer_parameters['w'] = init_parameters_w(i)
        parameters.append(layer_parameters)
    return parameters


parameters = init_parameters()


def predict(img, parameters):
    I0_in = img + parameters[0]['b']
    I0_out = activation[0](I0_in)
    I1_in = np.dot(I0_out, parameters[1]['w']) + parameters[1]['b']
    I1_out = activation[1](I1_in)
    return I1_out

print(predict(np.random.rand(784), parameters))
print(predict(np.random.rand(784), parameters).argmax())

