import numpy as np

class LLayerModel:
    """
    Train, test and predict using L-layer Deep Neural Network
    """

    def __init__(self, classes: str, num_iterations: int = 2000,
                 learning_rate: float = 0.0075, print_cost: bool = True):
        """ Initialize L-layer DNN with the provided layer_dims, classes, num_iterations, learning_rate and print_cost"""
        self.classes = classes
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.print_cost = print_cost
        self.costs = []

    def fit(self, layer_dims: tuple, train_set_x: np.ndarray, train_set_y: np.ndarray, test_set_x: np.ndarray, test_set_y: np.ndarray):
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        self.test_set_x = test_set_x
        self.test_set_y = test_set_y
        
        self.layer_dims = layer_dims    # start with n_x (num input features) & end with num of units on the output layer
        self.m = train_set_x.shape[1]  # no of training data

    def is_initialized(self):
        if not (self.train_set_x and self.train_set_y and self.test_set_x and self.test_set_y and self.layer_dims):
            return False
        return True
