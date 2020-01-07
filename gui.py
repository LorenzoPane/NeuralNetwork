import matplotlib.pyplot as plt
import NeuralNetwork as nn
import numpy as np

testing_images, testing_labels = nn.load_data('mnist/test.csv')

network = nn.Network([784, 100, 10], .5)
network.weights = np.load('dump.npy', allow_pickle=True)

plt.gray()

for i in range(len(testing_images)):
    image = testing_images[i]
    for a, b in enumerate(["{:.2f}%".format(num*100) for num in network.apply_network(testing_images[i])[:, 0]], 0):
        print(a, b)
    plt.imshow(image.reshape(28, 28))
    plt.show()
