import Object
import pandas as pd
import numpy as np

dataset = pd.read_csv("train.csv")
label = np.array(dataset.iloc[:, 0])
dataset = np.array(dataset)
dataset = np.delete(dataset, 0, axis=1)
dataset = dataset
dataset = dataset/255


def forward(layer1, layer2, input):
    layer1.forward(input)
    Activation1.forward(layer1.output)
    layer2.forward(Activation1.output)
    Activation2.forward(layer2.output)

    return layer1.output, Activation1.output, layer2.output, Activation2.output


def one_hot_y_observed(y_observed):
    array_temp = np.zeros((len(y_observed), np.max(y_observed) + 1))
    array_temp[np.arange(len(y_observed)), y_observed] = 1
    return array_temp


def deriv_relu(z):
    return z > 0


def backward(outputact2, outputlay2, outputact1, outputlay1, layer2, input, y_observed):
    m = len(y_observed)
    one_hot_y = one_hot_y_observed(y_observed)
    error2 = outputact2 - one_hot_y
    difference_weight_2 = 1 / m * error2.T.dot(outputact1)
    difference_biases_2 = 1 / m * np.sum(error2, axis=0, keepdims=True)
    error_1 = (layer2.weights.dot(error2.T)).T * deriv_relu(outputlay1)
    difference_weight_1 = 1 / m * error_1.T.dot(input)
    difference_biases_1 = 1 / m * np.sum(error_1, axis=0, keepdims=True)
    return (difference_weight_1.T, difference_biases_1, difference_weight_2.T,
            difference_biases_2)


def update_params(layer1, layer2, dw1, db1, dw2, db2, rate):
    layer1.weights = layer1.weights - rate * dw1
    layer1.biases = layer1.biases - rate * db1
    layer2.weights = layer2.weights - rate * dw2
    layer2.biases = layer2.biases - rate * db2


def get_predictions(A2):
    return np.argmax(A2, 1)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


layer_dense_1 = Object.Layer(784, 20)
Activation1 = Object.RectifiedLinearUnit()
layer_dense_2 = Object.Layer(20, 10)
Activation2 = Object.SoftMax()

for i in range(1000):
    start_indices = np.random.randint(0, 41000)
    temp_dataset = dataset[start_indices:start_indices+1000]
    temp_label = label[start_indices:start_indices+1000]
    outlay1, outact1, outlay2, outact2 = forward(layer_dense_1, layer_dense_2, temp_dataset)
    dw1, db1, dw2, db2 = backward(outact2, outlay2, outact1, outlay1, layer_dense_2, temp_dataset, temp_label)
    update_params(layer_dense_1, layer_dense_2, dw1, db1, dw2, db2, 0.2)

    print("\n")
    if i % 50 == 0:
        print("Iteration: ", i)
        predictions = get_predictions(outact2)
        print(get_accuracy(predictions, temp_label))
