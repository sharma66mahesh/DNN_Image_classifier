import numpy as np
from typing import Tuple, List


class LLayerModel:
    """
    Train, test and predict using L-layer Deep Neural Network
    """

    def __init__(self, classes: List[str, str], num_iterations: int = 2000,
                 learning_rate: float = 0.0075, print_cost: bool = True):
        """Initialize L-layer DNN with the provided layer_dims, classes, num_iterations, learning_rate and print_cost

        Args:
            classes (str): ["cat", "non-cat"]
            num_iterations (int, optional): Defaults to 2000.
            learning_rate (float, optional): Defaults to 0.0075.
            print_cost (bool, optional): Defaults to True.
        """
        self.classes = classes
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.print_cost = print_cost
        self.costs = []
        
        self.parameters = {}

    def fit(self, layer_dims: Tuple[int, ...], train_set_x: np.ndarray, train_set_y: np.ndarray,
            test_set_x: np.ndarray, test_set_y: np.ndarray):
        """Train the model params according to provided dataset and hyper-parameters(layer count & units in each layer)

        Args:
            layer_dims (Tuple[int, ...]): Starting with input features count and ending with num_units in the o/p layer
            train_set_x (np.ndarray): input features of train data set 
            train_set_y (np.ndarray): labels of the train data set
            test_set_x (np.ndarray): input features of test data set 
            test_set_y (np.ndarray): labels of the test data set
        """
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        self.test_set_x = test_set_x
        self.test_set_y = test_set_y

        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1
        self.m = train_set_x.shape[1]  # no. of training data
        
        # initialize params
        self.parameters = self.initialize_parameters(layer_dims)
        
        for i in range(self.num_iterations):
            # forward propagation
            pass
            # cost calculation
            pass
            # backward propagation
            pass
            # update model parameters
            pass

    def is_initialized(self) -> bool:
        """Check if the model is initialized with all the necessary hyperparameters and datasets

        Returns:
            bool: Bool status on the initialization status
        """
        if not (self.train_set_x and self.train_set_y and self.test_set_x and self.test_set_y and self.layer_dims):
            return False
        return True

    def initialize_parameters(layer_dims: Tuple[int, ...]):
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
