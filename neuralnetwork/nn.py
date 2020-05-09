from typing import List
import numpy as np
from neuralnetwork.activation import sigmoid


class NeuralNetwork(object):
    def __init__(self,
                 inputnodes: int,
                 hiddennodes: int,
                 outputnodes: int,
                 learningrate: float):
        # Shape
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Learning rate
        self.lr = learningrate

        # weights
        self.wih = np.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: sigmoid(x)

    def train(self):
        pass

    def query(self,
              inputs: List[float]):
        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)

        return self.activation_function(final_inputs)


def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    learning_rate = 0.3

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print(nn.query([1, 0.5, -1.5]))


if __name__ == "__main__":
    main()
