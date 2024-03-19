from utils.dnn_utils import load_dataset
from model.LLayerModel import LLayerModel
import matplotlib.pyplot as plt
import numpy as np

train_dataset_path = 'datasets/train_catvnoncat.h5'
test_dataset_path = 'datasets/test_catvnoncat.h5'

# load test and train data
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset(
    train_dataset_path=train_dataset_path, test_dataset_path=test_dataset_path)

# format train and test data
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# normalize data
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

dnn_model = LLayerModel(classes, num_iterations=2000, learning_rate=0.0075, print_cost=True)

# Train model
print("Training the model...\n\n")
four_layer_dims = (train_set_x.shape[0], 20, 7, 5, 1) # 4-LAYER
params, costs = dnn_model.fit(four_layer_dims, train_set_x, train_set_y, test_set_x, test_set_y)

# Make predictions on the test set and print accuracy
predictions, accuracy = dnn_model.predict(test_set_x, test_set_y)
print("Accuracy = " + str(accuracy * 100) + "%")

plt.plot(np.squeeze(costs))
plt.title("Cost per 100 iterations")
plt.xlabel("Iterations per 100")
plt.ylabel("Cost")
plt.show()