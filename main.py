import sys

import numpy as np


def initialize_parameters(layer_dims):
    # TODO This should be implemented as
    # {
    #     "W1": ...
    #     "b1": ...
    #     "W2": ...
    #     "b2": ...
    # }
    params_dict = {"w": [], "b": []}
    params_dict["w"].append(np.random.randn(1))
    params_dict["b"].append(np.zeros(1))
    for i in range(1, len(layer_dims)):
        params_dict["w"].append(np.random.randn(layer_dims[i], layer_dims[i - 1]))
        params_dict["b"].append(np.zeros(layer_dims[i]))
    return params_dict


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    linear_cache = {"A": A, "W": W, "b": b}
    return Z, linear_cache


def softmax(Z):
    sum_z = sum(np.exp(Z))
    A = [np.exp(z) / sum_z for z in Z]
    activation_cache = {"Z": Z}
    return A, activation_cache


def relu(Z):
    A = [max(0, z) for z in Z]
    activation_cache = {"Z": Z}
    return A, activation_cache


def linear_activation_forward(A_prev, W, B, activation):
    Z, linear_cache = linear_forward(A_prev, W, B)
    if activation == "softmax":
        A, activation_cache = softmax(Z)
    else:
        A, activation_cache = relu(Z)

    cache = dict(linear_cache)
    cache.update(activation_cache)
    return A, cache


def l_model_forward(X, parameters, use_batchnorm):
    caches = list()
    A = X
    L = len(parameters["w"])

    for layer_num in range(1, L):
        a_prev = A
        w = parameters["w"][layer_num]
        b = parameters["b"][layer_num]
        A, tmp_cache = linear_activation_forward(a_prev, w, b, activation='relu')

        caches.append(tmp_cache)

    w = parameters["w"][L]
    b = parameters["b"][L]
    AL, tmp_cache = linear_activation_forward(A, w, b, activation='sigmoid')
    caches.append(tmp_cache)

    return AL, caches


def compute_cost(AL, Y):
    # TODO
    return cost


def apply_batchnorm(A):
    epsilon = sys.float_info.epsilon
    mean = np.mean(A, axis=0)
    variance = np.var(A, axis=0)
    A_centered = A - mean

    NA = A_centered / np.sqrt(variance + epsilon)
    batchnorm_cache = {'activation': A, 'activation_norm': NA, 'mean': mean, 'var': variance}

    return NA, batchnorm_cache


def linear_backward(dZ, cache):
    A_prev = cache['A']
    W = cache['W']
    M = A_prev.shape[0]

    dA_prev = np.dot(W.T, dZ)
    dW = (1 / M) * dZ.dot(A_prev.T)
    db = (1 / M) * np.sum(dZ, axis=0, keepdims=True)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    else:
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def relu_backward(dA, activation_cache):
    Z = activation_cache['Z']
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def softmax_backward(dA, activation_cache):
    Z = activation_cache
    Z = Z - np.max(Z)
    sum = (np.exp(Z).T / np.sum(np.exp(Z), axis=1))
    dZ = dA * sum * (1 - sum)

    return dZ


def l_model_backward(AL, Y, caches):
    # TODO
    grads = dict()

    return grads


def update_parameters(parameters, grads, learning_rate):
    # TODO
    return parameters


def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    # TODO
    parameters = None
    costs = None

    return parameters, costs


def predict(X, Y, parameters):
    # TODO
    accuracy = None
    return accuracy


if __name__ == '__main__':
    result = initialize_parameters([3, 4, 1])
    print(result)
