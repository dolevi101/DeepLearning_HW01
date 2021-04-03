import sys

import keras
import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def initialize_parameters(layer_dims):
    params_dict = {"w": [], "b": []}
    params_dict["w"].append(np.random.randn(1))
    params_dict["b"].append(np.zeros(1))
    for i in range(1, len(layer_dims)):
        params_dict["w"].append(np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2 / layer_dims[0]))
        params_dict["b"].append(np.zeros((layer_dims[i], 1)))
    return params_dict


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    linear_cache = {"A": A, "W": W, "b": b}
    return Z, linear_cache


def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    activation_cache = {"Z": Z}
    return A, activation_cache


def relu(Z):
    A = np.maximum(0, Z)
    activation_cache = {"Z": Z}
    return A, activation_cache


def linear_activation_forward(A_prev, W, B, activation, use_batchnorm):
    Z, linear_cache = linear_forward(A_prev, W, B)

    if use_batchnorm:
        Z = apply_batchnorm(Z)

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
    L = len(parameters["w"]) - 1

    for layer_num in range(1, L):
        if use_batchnorm:
            A = apply_batchnorm(A)

        a_prev = A
        w = parameters["w"][layer_num]
        b = parameters["b"][layer_num]
        A, tmp_cache = linear_activation_forward(a_prev, w, b, activation='relu', use_batchnorm=use_batchnorm)

        caches.append(tmp_cache)

    if use_batchnorm:
        A = apply_batchnorm(A)

    w = parameters["w"][L]
    b = parameters["b"][L]
    AL, tmp_cache = linear_activation_forward(A, w, b, activation='softmax', use_batchnorm=False)
    caches.append(tmp_cache)

    return AL, caches


def compute_cost(AL, Y):
    num_examples = AL.shape[1]
    cost = (-1 / num_examples) * np.sum(np.multiply(Y, np.log(AL)))

    return cost


def apply_batchnorm(A):
    epsilon = sys.float_info.epsilon
    mean = np.mean(A, axis=0)
    variance = np.var(A, axis=0)
    A_centered = A - mean

    NA = A_centered / np.sqrt(variance + epsilon)

    return NA


def linear_backward(dZ, cache):
    A_prev = cache['A']
    W = cache['W']
    num_examples = A_prev.shape[1]

    dA_prev = np.dot(W.T, dZ)
    dW = (1 / num_examples) * dZ.dot(A_prev.T)
    db = (1 / num_examples) * np.sum(dZ, axis=1, keepdims=True)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache = dict()
    activation_cache = dict()
    linear_cache["A"] = cache["A"]
    linear_cache["W"] = cache["W"]
    linear_cache["b"] = cache["b"]
    activation_cache["Z"] = cache["Z"]
    activation_cache["Y"] = cache["Y"]

    if activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)

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
    Z = activation_cache["Z"]
    Y = activation_cache["Y"]
    A, cache = softmax(Z)

    dZ = A - Y

    return dZ


def l_model_backward(AL, Y, caches):
    grads = dict()
    num_layers = len(caches)

    dAL = ((-1) * (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)))

    last_layer_cache = caches[num_layers - 1]
    last_layer_cache["Y"] = Y
    (dA_prev_temp,
     dW_temp,
     db_temp) = linear_activation_backward(dAL, last_layer_cache, activation="softmax")

    tmp_grads = {"dA" + str(num_layers): dA_prev_temp,
                 "dW" + str(num_layers): dW_temp,
                 "db" + str(num_layers): db_temp}
    grads.update(tmp_grads)

    for layer_number in reversed(range(1, num_layers)):
        tmp_cache = caches[layer_number - 1]
        tmp_cache["Y"] = Y
        (dA_prev_temp,
         dW_temp,
         db_temp) = linear_activation_backward(grads["dA" + str(layer_number + 1)], tmp_cache, activation="relu")

        tmp_grads = {"dA" + str(layer_number): dA_prev_temp,
                     "dW" + str(layer_number): dW_temp,
                     "db" + str(layer_number): db_temp}
        grads.update(tmp_grads)

    return grads


def update_parameters(parameters, grads, learning_rate):
    num_of_layers = len(parameters["w"]) - 1

    for layer_num in range(1, num_of_layers + 1):
        parameters["w"][layer_num] = parameters["w"][layer_num] - (learning_rate * grads["dW" + str(layer_num)])
        parameters["b"][layer_num] = parameters["b"][layer_num] - (learning_rate * grads["db" + str(layer_num)])

    return parameters


def next_batch(X, Y, batch_size=1):
    num_of_examples = len(X)
    for next_batch_idx in range(0, num_of_examples, batch_size):
        yield (X[:, next_batch_idx:min(next_batch_idx + batch_size, num_of_examples)],
               Y[:, next_batch_idx:min(next_batch_idx + batch_size, num_of_examples)])


def split_data(X, Y):
    X_train, X_val, y_train, y_val = train_test_split(X.T, Y.T,
                                                      test_size=0.2,
                                                      stratify=Y.T, random_state=42)
    X_train, X_val, y_train, y_val = X_train.T, X_val.T, y_train.T, y_val.T
    return X_train, X_val, y_train, y_val


def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm, min_epochs):
    X_train, X_val, y_train, y_val = split_data(X, Y)

    parameters = initialize_parameters([X.shape[0]] + layers_dims)

    iteration_counter = 0
    best_validation_acc = 0
    validation_acc_not_improved = 0
    parameter_saver = list()

    for epoch in range(min_epochs):
        for X_batch, Y_batch in next_batch(X_train, y_train, batch_size):
            AL, caches = l_model_forward(X_batch, parameters, use_batchnorm)
            cost = compute_cost(AL, Y_batch)
            gradients = l_model_backward(AL, Y_batch, caches)
            parameters = update_parameters(parameters, gradients, learning_rate)

            validation_acc = predict(X_val, y_val, parameters)
            #train_acc = predict(X_train, y_train, parameters)

            parameter_saver.append(
                {"iteration": iteration_counter,
                 "epoch": epoch,
                 "cost": cost,
                 "validation_acc": validation_acc})

            if iteration_counter % 100 == 0:
                print(f'iteration: {iteration_counter} | cost: {cost} | val_acc: {validation_acc}')

            if validation_acc > best_validation_acc:
                best_validation_acc = validation_acc
                validation_acc_not_improved = 0
            else:
                validation_acc_not_improved += 1

            if validation_acc_not_improved >= 100:
                return parameters, parameter_saver

            iteration_counter += 1

    return parameters, parameter_saver


def predict(X, Y, parameters):
    scores, _ = l_model_forward(X, parameters, use_batchnorm=False)  # test time
    predictions = np.argmax(scores, axis=0)
    Y_flatten = np.argmax(Y, axis=0)
    return accuracy_score(Y_flatten, predictions)


def print_shapes(x_train, y_train, x_test, y_test):
    print(f"x_train.shape = {x_train.shape}")
    print(f"y_train.shape = {y_train.shape}\n")
    print(f"x_test.shape = {x_test.shape}")
    print(f"y_test.shape = {y_test.shape}\n")


def prepare_data(x_train, y_train, x_test, y_test):
    """
    Perform one-hot encoding to the labels, and reshaping to the data
    """
    num_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train.reshape(-1, 1), num_classes)
    y_test = keras.utils.to_categorical(y_test.reshape(-1, 1), num_classes)

    image_size = 784
    x_train = x_train.reshape(-1, image_size)
    x_test = x_test.reshape(-1, image_size)

    return x_train, y_train, x_test, y_test


def plot(to_plot, title='Title', xlabel='', ylabel=''):
    import matplotlib.pyplot as plt
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(to_plot)
    plt.show()


if __name__ == '__main__':
    # result = initialize_parameters([3, 4, 1])
    # print(result)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print_shapes(X_train, y_train, X_test, y_test)
    print("classes = ", list(np.unique(y_train)))

    # data preprocessing
    X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test)
    print_shapes(X_train, y_train, X_test, y_test)
    # X_train, X_test = scale_data(X_train, X_test)

    hidden_dims = [20, 7, 5, 10]
    lr = 0.009
    iters = 9000000
    min_epochs = 50

    # batch sizes experiment
    batch_experiment_results = {}
    for batch_size in [32, 64, 128]:
        batch_experiment_results[batch_size] = l_layer_model(X_train.T, y_train.T,
                                                             hidden_dims,
                                                             learning_rate=lr,
                                                             batch_size=batch_size,
                                                             use_batchnorm=False,
                                                             num_iterations=iters,
                                                             min_epochs=min_epochs)
