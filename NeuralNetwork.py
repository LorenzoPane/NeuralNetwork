import numpy as np
from scipy.stats import truncnorm


# from https://stackoverflow.com/a/42874900
def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def sigmoid(x):
    return 1 / (1 + np.e ** -x)


class Network:
    def __init__(self, in_size, hidden_size, out_size, bias):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.bias = bias

        # Init random weights
        half_range = 1 / np.sqrt(in_size + 1)
        tn = truncnorm(-half_range, half_range, scale=1, loc=0)
        self.input_to_hidden_weights = tn.rvs((hidden_size, in_size + 1))

        half_range = 1 / np.sqrt(hidden_size + 1)
        tn = truncnorm(-half_range, half_range, scale=1, loc=0)
        self.hidden_to_output_weights = tn.rvs((out_size, hidden_size + 1))

    def train(self, data, labels, iterations, step):
        for iteration in range(iterations):
            for index in range(len(data)):
                in_v = np.array(np.concatenate((data[index], [self.bias])), ndmin=2).T
                target_v = np.array(labels[index], ndmin=2).T

                # Apply network
                hidden_output = np.concatenate((sigmoid(np.dot(self.input_to_hidden_weights, in_v)), [[self.bias]]))
                output_output = sigmoid(np.dot(self.hidden_to_output_weights, hidden_output))

                # Adjust weights
                diff = target_v - output_output
                self.hidden_to_output_weights += step * np.dot(diff * output_output * (1 - output_output), hidden_output.T)

                diff = np.dot(self.hidden_to_output_weights.T, diff)
                self.input_to_hidden_weights += step * np.dot(diff * hidden_output * (1 - hidden_output), in_v.T)[:-1, :]

    def apply_network(self, in_v):
        in_v = np.array(np.concatenate((in_v, [self.bias])), ndmin=2).T
        hidden_output = np.concatenate((sigmoid(np.dot(self.input_to_hidden_weights, in_v)), [[self.bias]]))
        return sigmoid(np.dot(self.hidden_to_output_weights, hidden_output))

    def test(self, data, labels):
        success, failure = 0, 0

        for i in range(len(data)):
            result = self.apply_network(data[i]).argmax()
            if result == labels[i].argmax():
                success += 1
            else:
                failure += 1

        return success, failure
