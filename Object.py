import numpy as np
import math


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = abs(0.1 * np.random.randn(n_inputs, n_neurons))
        self.biases = abs(0.1 * np.random.randn(1, n_neurons))
        self.output = []

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class RectifiedLinearUnit:
    def __init__(self):
        self.output = 0

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class SoftMax:
    def __init__(self):
        self.output = 0

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


class Loss:
    def calculate(self, output, y_observed):
        sample_loss = self.forward(output, y_observed)
        data_loss = np.mean(sample_loss)
        return data_loss


class CategoricalLossEntropy(Loss):
    def forward(self, y_pred, y_observed):
        samples = len(y_pred)

        if len(y_observed.shape) == 1:
            correct_confidences = y_pred[range(samples), y_observed]

        elif len(y_observed.shape) == 2:
            correct_confidences = np.sum((y_pred*y_observed), axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

