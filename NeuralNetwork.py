import numpy as np
from typing import Iterator


Vector = np.array
Matrix = np.array


# from https://stackoverflow.com/a/42874900
def indices_to_one_hot(data: int, nb_classes: int) -> Vector:
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def load_data(path: str) -> (np.array, Iterator):
    raw = np.loadtxt(path, delimiter=',')

    labels = (int(i) for i in raw[:, 0])
    labels = (indices_to_one_hot(l, 10) for l in labels)

    images = np.asfarray(raw)[:, 1:] * 0.99 / 255 + 0.01

    return images, labels


def sigmoid(x):
    return 1 / (1 + np.e ** -x)


class Network:
    def __init__(self, structure: list, bias: float):
        self.bias = bias
        self.weights = []

        # Init random weights
        for in_dimensions, out_dimensions in zip(structure, structure[1:]):
            scale = 1 / 3 / np.sqrt(in_dimensions + 1)
            self.weights.append(
                np.random.normal(0, scale, (out_dimensions, in_dimensions + 1)))

    @staticmethod
    def apply_layer(v: Vector, weight_matrix: Matrix) -> Vector:
        return sigmoid(np.dot(weight_matrix, v))

    def add_bias(self, v: Vector) -> Vector:
        return np.concatenate((v, [[self.bias]]))

    def apply_network(self, v: Vector) -> Vector:
        v = np.array(v, ndmin=2).T

        for layer in self.weights:
            v = self.apply_layer(self.add_bias(v), layer)

        return v

    def train(self, data: Iterator, labels: Iterator, iterations=1, step=0.1):
        for _ in range(iterations):
            for in_v, target_v in zip(data, labels):
                # Apply
                vs = [np.array(in_v, ndmin=2).T]
                for layer in self.weights:
                    vs.append(self.apply_layer(self.add_bias(vs[-1]), layer))

                # Adjust
                diff = target_v.T - vs[-1]
                for i in reversed(range(len(self.weights))):
                    self.weights[i] += step * np.dot(
                        diff * vs[i+1] * (1. - vs[i+1]),
                        self.add_bias(vs[i]).T)
                    diff = np.dot(self.weights[i].T[:-1], diff)

    def test(self, images: Iterator, labels: Iterator) -> (int, int):
        success, failure = 0, 0

        for image, label in zip(images, labels):
            result = self.apply_network(image).argmax()
            if result == label.argmax():
                success += 1
            else:
                failure += 1

        return success, failure
