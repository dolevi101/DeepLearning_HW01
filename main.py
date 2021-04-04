import datetime
import sys

import keras
import numpy as np
from keras.datasets import mnist


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


def l_model_forward(X, parameters, use_batchnorm, dropout_prob=None):
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

        if dropout_prob:
            A, dropout_mask = dropout_forward(A, dropout_prob)
            tmp_cache.update({'dropout_mask': dropout_mask})

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
        dA_temp = grads["dA" + str(layer_number + 1)]

        if 'dropout_mask' in tmp_cache:
            dA_temp = dropout_backward(dA_temp, tmp_cache['dropout_mask'])

        tmp_cache["Y"] = Y
        (dA_prev_temp,
         dW_temp,
         db_temp) = linear_activation_backward(dA_temp, tmp_cache, activation="relu")

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


def dropout_forward(X, prob):
    """
    Applies the forward dropout function
    @param X: The activations to apply the dropout on
    @param prob: The probability to apply the dropout
    @return:
        out - The activations after the dropout
        mask - The mask used for dropout, in order to cache
    """
    mask = (np.random.rand(*X.shape) < prob) / prob
    out = X * mask
    return out, mask


def dropout_backward(dX, mask):
    """
    Applies the backward dropout function
    @param dX: The derivative to apply the dropout on
    @param mask: The cached mask from the forward dropout process
    @return: The dropped-out activation
    """
    dX = dX * mask
    return dX


def next_batch(X, Y, batch_size):
    """
    Generates batches for X and Y sets
    @param X: The training values
    @param Y: The real target values
    @param batch_size: The batch size to use
    @return: Touple of (X,Y) of the next batch
    """
    num_of_examples = X.shape[1]
    for next_batch_idx in range(0, num_of_examples, batch_size):
        yield (X[:, next_batch_idx:min(next_batch_idx + batch_size, num_of_examples)],
               Y[:, next_batch_idx:min(next_batch_idx + batch_size, num_of_examples)])


def split_data(X, Y):
    """
    Creates a train/val split for the data
    @param X: The train values
    @param Y: The target values
    @return:
        X_train, X_val, y_train, y_val - The values after the split
    """
    msk = np.random.rand(X.shape[1]) < 0.8

    X_train = X[:, msk]
    X_val = X[:, ~msk]

    y_train = Y[:, msk]
    y_val = Y[:, ~msk]

    return X_train, X_val, y_train, y_val


def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False, dropout_prob=None):
    X_train, X_val, y_train, y_val = split_data(X, Y)

    parameters = initialize_parameters([X.shape[0]] + layers_dims)

    iteration_counter = 0
    epoch_counter = 0
    best_validation_acc = 0
    validation_acc_not_improved = 0
    cost_parameter_saver = list()

    while iteration_counter < num_iterations:
        for X_batch, Y_batch in next_batch(X_train, y_train, batch_size):
            AL, caches = l_model_forward(X_batch, parameters, use_batchnorm, dropout_prob)
            cost = compute_cost(AL, Y_batch)
            gradients = l_model_backward(AL, Y_batch, caches)
            parameters = update_parameters(parameters, gradients, learning_rate)

            validation_acc = predict(X_val, y_val, parameters)

            if iteration_counter % 100 == 0:
                print(
                    f'iteration: {iteration_counter} | cost: {cost} | val_acc: {validation_acc} | epoch: {epoch_counter}')
                cost_parameter_saver.append(cost)

            if validation_acc > best_validation_acc:
                best_validation_acc = validation_acc
                validation_acc_not_improved = 0
            else:
                validation_acc_not_improved += 1

            if validation_acc_not_improved >= 100 or iteration_counter >= num_iterations:
                return parameters, cost_parameter_saver

            iteration_counter += 1

        epoch_counter += 1
    return parameters, cost_parameter_saver


def predict(X, Y, parameters):
    predictions, _ = l_model_forward(X, parameters, use_batchnorm=False, dropout_prob=None)
    predictions_labeled = np.argmax(predictions, axis=0)  # converting one-hot vectors to only chosen-label vector

    Y_labeled = np.argmax(Y, axis=0)
    temp = predictions_labeled - Y_labeled
    nonzero_vals = np.count_nonzero(temp)

    accuracy = (1 - nonzero_vals / Y_labeled.shape[0])

    return accuracy


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    num_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train.reshape(-1, 1), num_classes)
    y_test = keras.utils.to_categorical(y_test.reshape(-1, 1), num_classes)

    image_size = 784
    X_train = X_train.reshape(-1, image_size)
    X_test = X_test.reshape(-1, image_size)

    hidden_layer_dims = [20, 7, 5, 10]
    lr = 0.009
    num_iterations = 100000
    tested_batch_sizes = [32, 64, 128]

    results = list()
    for batch_size in tested_batch_sizes:
        start_time = datetime.datetime.now()
        parameters, cost_parameter_saver = l_layer_model(X_train.T, y_train.T,
                                                         hidden_layer_dims,
                                                         learning_rate=lr,
                                                         batch_size=batch_size,
                                                         num_iterations=num_iterations,
                                                         use_batchnorm=False,
                                                         dropout_prob=None)

        end_time = datetime.datetime.now()

        last_parameters = cost_parameter_saver[-1]
        results_dict = {"time": str(end_time - start_time),
                        "batch_size": batch_size,
                        "iterations": len(cost_parameter_saver) * 100,
                        "cost": cost_parameter_saver[-1],
                        "train_acc": predict(X_train.T, y_train.T, parameters),
                        "test_acc": predict(X_test.T, y_test.T, parameters)}

        print(results_dict)

        results.append(results_dict)

    print(results)
