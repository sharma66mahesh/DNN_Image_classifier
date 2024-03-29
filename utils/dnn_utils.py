import h5py
import numpy as np
from typing import Tuple, Dict


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- Z; stored for computing the backward pass efficiently
    """
    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dZ to a correct object

    # =============NOTE::================
    # g'(z) = 0 if z <=0, else g'(z) = 1 ===> for ReLu
    # dz = da * g'(z)

    # when z <=0, dz = 0
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache

    s = 1 / (1 + np.exp(Z))
    # dz = da * g'(z), where g'(z) = g(z) * (1 - g(z)) ===> for sigmoid
    dZ = dA * s * (1 - s + 10 ** -8)

    assert (dZ.shape == Z.shape)

    return dZ


def initialize_parameters(layer_dims: Tuple[int, ...]) -> Dict:
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network. Eg: [n_x, 4,3,2,1], where 1st layer has 4 neurons, 2nd layer has 3 neurons, and so on...

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],
                                                   layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape ==
                (layer_dims[l], layer_dims[l-1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

# Layer XYZ to Layer XYZ + 1


def linear_forward(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = W.dot(A_prev) + b
    assert (Z.shape == (W.shape[0], A_prev.shape[1]))

    cache = (A_prev, W, b)
    return Z, cache

# Layer XYZ to Layer XYZ + 1


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)

    assert (Z.shape == (W.shape[0], A_prev.shape[1]))
    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)
    return A, cache

# Traverse the whole DNN (Deep Neural Network)


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    L = len(parameters) // 2    # /2 since w1 & b1 (for example) belong to the same layer
    A_prev = X

    caches = []

    for l in range(1, L):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]

        A, cache = linear_activation_forward(A_prev, W, b, "relu")

        caches.append(cache)
        A_prev = A

    AL, cache = linear_activation_forward(
        A_prev, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))     # 1 * m

    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cross-entropy cost function

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]

    cost = -1 / m * (np.dot(Y, np.log(AL).T) + np.dot((1 - Y), np.log(1-AL).T))

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost


# linear portion of backward propagation for a single layer
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * (np.dot(dZ, A_prev.T))
    db = 1 / m * (np.sum(dZ, axis=1, keepdims=True))
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

# Compute dZ


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches)
    dAL = -1 * (Y / AL) + (1 - Y) / (1 - AL)

    dA_prev, dW, db = linear_activation_backward(
        dA=dAL, cache=caches[L-1], activation="sigmoid")
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" +
                                                      str(L)] = (dA_prev, dW, db)

    for l in reversed(range(1, L)):
        dA = dA_prev
        dA_prev, dW, db = linear_activation_backward(
            dA=dA, cache=caches[l-1], activation="relu")
        grads["dA" + str(l)], grads["dW" + str(l)
                                    ], grads["db" + str(l)] = (dA_prev, dW, db)

    return grads

# Learn and update parameters


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    L = len(parameters) // 2  # /2 since there is W & b for each layer

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - \
            grads["dW" + str(l+1)] * learning_rate
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - \
            grads["db" + str(l+1)] * learning_rate

    return parameters


def load_dataset(train_dataset_path: str, test_dataset_path: str):
    train_dataset = h5py.File(train_dataset_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File(test_dataset_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y = train_set_y_orig.reshape(1, train_set_y_orig.shape[0])
    test_set_y = test_set_y_orig.reshape(1, test_set_y_orig.shape[0])

    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes
