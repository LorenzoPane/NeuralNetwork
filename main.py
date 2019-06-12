import numpy as np
import NeuralNetwork as nn

# load raw data in format (https://pjreddie.com/projects/mnist-in-csv/)
raw_training = np.loadtxt('mnist/train.csv', delimiter=',')
raw_testing = np.loadtxt('mnist/test.csv', delimiter=',')

# load labels
training_labels = np.array([int(i) for i in raw_training[:, 0]])
testing_labels = np.array([int(i) for i in np.array(raw_testing)[:, 0]])

# one hot representations of labels
training_labels = [nn.indices_to_one_hot(l, 10) for l in training_labels]
testing_labels = [nn.indices_to_one_hot(l, 10) for l in testing_labels]

# load images
training_images = np.asfarray(raw_training)[:, 1:]
testing_images = np.asfarray(raw_testing)[:, 1:]

# scale pixels from the interval [0, 255] to [0.01, 1]
training_images = training_images * 0.99 / 255 + 0.01
testing_images = testing_images * 0.99 / 255 + 0.01

network = nn.Network(784, 100, 10, .5)
network.train(training_images, training_labels, 1, 0.1)
correct, wrong = network.test(testing_images, testing_labels)
print(float(correct)/100)

np.savetxt('dumped_network/input_to_hidden.csv', network.input_to_hidden_weights)
np.savetxt('dumped_network/hidden_to_output.csv', network.hidden_to_output_weights)
