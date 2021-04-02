import sys

import keras
import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
        params_dict["b"].append(np.zeros((layer_dims[i], 1)))
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
    A = np.maximum(0,Z)
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
        if use_batchnorm:
            A = apply_batchnorm(A)

        a_prev = A
        w = parameters["w"][layer_num]
        b = parameters["b"][layer_num]
        A, tmp_cache = linear_activation_forward(a_prev, w, b, activation='relu')

        caches.append(tmp_cache)

    if use_batchnorm:
        A = apply_batchnorm(A)

    w = parameters["w"][L]
    b = parameters["b"][L]
    AL, tmp_cache = linear_activation_forward(A, w, b, activation='softmax')
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

    return NA


def linear_backward(dZ, cache):
    A_prev = cache['A_prev']
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
    grads = {}
    num_layers = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    last_layer_cache = caches[num_layers - 1]
    (grads["dA" + str(num_layers)],
     grads["dW" + str(num_layers)],
     grads["db" + str(num_layers)]) = linear_activation_backward(dAL, last_layer_cache, activation="softmax")

    for layer in reversed(range(1, num_layers)):
        tmp_cache = caches[layer]
        (dA_prev_temp,
         dW_temp,
         db_temp) = linear_activation_backward(grads["dA" + str(layer + 1)], tmp_cache, activation="relu")

        grads["dA" + str(layer)] = dA_prev_temp
        grads["dW" + str(layer)] = dW_temp
        grads["db" + str(layer)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    num_of_layers = len(parameters) // 2

    for layer_num in range(1, num_of_layers):
        curr_w = "w" + str(layer_num)
        curr_b = "w" + str(layer_num)

        parameters[curr_w] -= (learning_rate * grads["dW" + str(layer_num)])
        parameters[curr_b] -= (learning_rate * grads["db" + str(layer_num)])

    return parameters


def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm, min_epochs):
    def next_batch(X, y, batch_size):
        # loop over our dataset X in mini-batches of size batchSize
        for i in np.arange(0, X.shape[0], batch_size):
            # yield a tuple of the current batched data and labels
            yield (X[:, i: i + batch_size], y[:, i: i + batch_size])

    # split to train and val
    X_train, X_val, y_train, y_val = train_test_split(X.T, Y.T,
                                                      test_size=0.2,
                                                      stratify=Y.T, random_state=42)

    X_train, X_val, y_train, y_val = X_train.T, X_val.T, y_train.T, y_val.T

    # initialization
    parameters = initialize_parameters([X.shape[0]] + layers_dims)
    costs = []
    accs_per_100_iterations = []
    costs_per_100_iterations = []
    train_accs_pre_100_iterations = []

    iterations_counter = 0
    epoch_counter = 0
    val_acc_no_improvement_count = 0
    best_val_acc_value = 0

    while iterations_counter < num_iterations:
        for X_batch, Y_batch in next_batch(X_train, y_train, batch_size):
            # forward pass
            AL, caches = l_model_forward(X_batch, parameters, use_batchnorm)

            # compute the cost and document it
            cost = compute_cost(AL, Y_batch)
            costs.append(cost)

            # backward pass
            grads = l_model_backward(AL, Y_batch, caches)

            # update parameters
            parameters = update_parameters(parameters, grads, learning_rate)

            iterations_counter += 1

            # document performance every 100 iterations
            val_acc = predict(X_val, y_val, parameters)
            if iterations_counter % 100 == 0:
                accs_per_100_iterations.append(val_acc)
                train_acc = predict(X_train, y_train, parameters)
                train_accs_pre_100_iterations.append(train_acc)
                costs_per_100_iterations.append(cost)
                print('iteration step: {} | cost: {}'.format(iterations_counter, cost))

            # check if accuracy improved
            if val_acc > best_val_acc_value:
                best_val_acc_value = val_acc
                val_acc_no_improvement_count = 0
            val_acc_no_improvement_count += 1

            # check stop criteria
            if val_acc_no_improvement_count >= 100 and epoch_counter >= min_epochs:
                train_acc = predict(X_train, y_train, parameters)
                val_acc = predict(X_val, y_val, parameters)
                return parameters, costs_per_100_iterations, accs_per_100_iterations, train_accs_pre_100_iterations, train_acc, val_acc
        epoch_counter += 1

    train_acc = predict(X_train, y_train, parameters)
    val_acc = predict(X_val, y_val, parameters)
    return parameters, costs_per_100_iterations, accs_per_100_iterations, train_accs_pre_100_iterations, train_acc, val_acc


def predict(X, Y, parameters):
    scores, _, _, _ = l_model_forward(X, parameters, use_batchnorm=False)  # test time
    predictions = np.argmax(scores, axis=1)
    Y_flatten = np.argmax(Y, axis=1)
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
    x_train = x_train.reshape(image_size, -1)
    x_test = x_test.reshape(image_size, -1)
    y_train = y_train.T
    y_test = y_test.T

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
    batch_size = 64
    iters = 9000000
    min_epochs = 10

    # batch sizes experiment
    batch_experiment_results = {}
    for batch_size in [32, 64, 128]:
        batch_experiment_results[batch_size] = l_layer_model(X_train, y_train,
                                                             hidden_dims,
                                                             learning_rate=lr,
                                                             batch_size=batch_size,
                                                             use_batchnorm=False,
                                                             num_iterations=iters,
                                                             min_epochs=min_epochs)
