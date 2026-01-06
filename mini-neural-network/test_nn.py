import numpy as np
from neural_network import MiniNeuralNetwork

def generate_sample_dataset():
    """
    Generates a simple synthetic dataset for binary classification.
    The task is to determine if a point (x, y) is inside or outside a circle.
    """
    np.random.seed(42)
    X = np.random.rand(200, 2) * 2 - 1  # 200 points between -1 and 1
    # Target: 1 if inside circle of radius 0.7, else 0
    y = (np.sum(X**2, axis=1) < 0.49).astype(int).reshape(-1, 1)
    return X, y

def main():
    print("--- Mini Neural Network Test ---")
    
    # 1. Generate Dataset
    X, y = generate_sample_dataset()
    print(f"Generated dataset with {len(X)} samples.")
    
    # 2. Split into Train and Test
    train_size = 160
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 3. Initialize and Train Network
    # Input: 2 (x, y coordinates), Hidden: 8 neurons, Output: 1 (probability)
    nn = MiniNeuralNetwork(input_size=2, hidden_size=8, output_size=1, learning_rate=0.1)
    
    print("\nStarting training...")
    nn.train(X_train, y_train, epochs=20000)
    
    # 4. Evaluate
    print("\nEvaluating on test set...")
    predictions = nn.forward(X_test)
    binary_predictions = (predictions > 0.5).astype(int)
    accuracy = np.mean(binary_predictions == y_test)
    
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # 5. Show some sample results
    print("\nSample Predictions (First 5 test samples):")
    for i in range(5):
        print(f"Input: {X_test[i]}, Actual: {y_test[i][0]}, Predicted: {predictions[i][0]:.4f} ({binary_predictions[i][0]})")

if __name__ == "__main__":
    main()
