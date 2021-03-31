import numpy as np


def initialize_parameters(layer_dims):
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
    # TODO
    cache = None
    AL = None

    return AL, cache


def compute_cost(AL, Y):
    # TODO
    cost = None
    return cost


def apply_batchnorm(A):
    # TODO
    NA = None

    return NA


def linear_backward(dZ, cache):
    # TODO
    dA_prev = None
    dW = None
    db = None

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    # TODO
    dA_prev = None
    dW = None
    db = None

    return dA_prev, dW, db


def relu_backward(dA, activation_cache):
    # TODO
    dZ = None

    return dZ


def softmax_backward(dA, activation_cache):
    # TODO
    dZ = None

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
