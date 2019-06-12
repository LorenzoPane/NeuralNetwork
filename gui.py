import matplotlib.pyplot as plt
import NeuralNetwork as nn
import numpy as np

raw_testing = np.loadtxt('mnist/test.csv', delimiter=',')
testing_labels = np.array([int(i) for i in np.array(raw_testing)[:, 0]])
testing_labels = [nn.indices_to_one_hot(l, 10) for l in testing_labels]
testing_images = np.asfarray(raw_testing)[:, 1:]
testing_images = testing_images * 0.99 / 255 + 0.01

network = nn.Network(784, 100, 10, .5)

network.input_to_hidden_weights = np.loadtxt('dumped_network/input_to_hidden.csv')
network.hidden_to_output_weights = np.loadtxt('dumped_network/hidden_to_output.csv')

plt.gray()

for i in range(len(testing_images)):
    image = testing_images[i]
    for a, b in enumerate(["{:.2f}%".format(num*100) for num in network.apply_network(testing_images[i])[:, 0]], 0):
        print(a, b)
    plt.imshow(image.reshape(28, 28))
    plt.show()
