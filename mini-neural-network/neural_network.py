import numpy as np


class MiniNeuralNetwork:
    """
    A simple 2-layer neural network (Input -> Hidden -> Output).
    This implementation uses the Sigmoid activation function and Mean Squared Error loss.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights with small random values
        # Weights connect layers: W1 connects Input to Hidden, W2 connects Hidden to Output
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        self.learning_rate = learning_rate

    def sigmoid(self, x):
        """The sigmoid activation function: maps any value to a range between 0 and 1."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function, used during backpropagation."""
        return x * (1 - x)

    def forward(self, X):
        """
        Forward pass: Calculate the network's output for a given input X.
        """
        # Layer 1 (Hidden Layer)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Layer 2 (Output Layer)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, output):
        """
        Backward pass: Calculate gradients and update weights/biases.
        """
        # Calculate error at the output layer
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        # Calculate error at the hidden layer
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        # Update weights and biases using gradient descent
        self.W2 += self.a1.T.dot(output_delta) * self.learning_rate
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

        self.W1 += X.T.dot(hidden_delta) * self.learning_rate
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000):
        """
        Train the neural network for a specified number of epochs.
        """
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.6f}")


if __name__ == "__main__":
    # Example usage with XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = MiniNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    print("Training the network on XOR problem...")
    nn.train(X, y, epochs=10000)

    print("\nPredictions after training:")
    for i in range(len(X)):
        pred = nn.forward(X[i : i + 1])
        print(f"Input: {X[i]}, Target: {y[i]}, Prediction: {pred[0][0]:.4f}")
