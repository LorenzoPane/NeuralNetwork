import numpy as np
import NeuralNetwork as nn


if __name__ == '__main__':
    training_images, training_labels = nn.load_data('mnist/train.csv')
    testing_images, testing_labels = nn.load_data('mnist/test.csv')

    network = nn.Network([784, 100, 10], .5)
    network.train(training_images, training_labels, 5, 0.01)

    correct, incorrect = network.test(testing_images, testing_labels)
    print(float(correct)/100)

    np.save('dump.npy', np.array(network.weights))
