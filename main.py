import numpy as np
from numpy import random


def initialize_parameters(layer_dims):
    params_dict = {"w": [], "b": []}
    params_dict["w"].append(np.random.rand(1))
    params_dict["b"].append(np.zeros(1))
    for i in range(1, len(layer_dims)):
        params_dict["w"].append(np.random.rand(layer_dims[i], layer_dims[i - 1]))
        params_dict["b"].append(np.zeros(layer_dims[i]))
    return params_dict


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    linear_cache = {"A": A, "W": W, "b": b}
    return Z, linear_cache


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    result = initialize_parameters([3, 4, 1])
    print(result["w"])
    print()
    print(result["b"])
