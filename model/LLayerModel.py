import numpy as np
from typing import Tuple, List
from utils.dnn_utils import *


class LLayerModel:
    """
    Train, test and predict using L-layer Deep Neural Network
    """

    def __init__(self, classes: List[str], num_iterations: int = 2000,
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

        # ensure that the model has all the data required
        if not self.is_initialized():
            raise Exception(
                "Model might be missing train/test datasets or layer_dims")

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
            bool: Bool value of the initialization status
        """
        if not (len(self.train_set_x) > 0 and len(self.train_set_y) > 0 and len(self.test_set_x) > 0
                and len(self.test_set_y) > 0 and len(self.layer_dims) > 0):
            return False
        return True
