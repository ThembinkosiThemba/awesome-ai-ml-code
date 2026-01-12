import numpy as np
from decision_tree import DecisionTree

def generate_dataset():
    """
    Generates a synthetic dataset for binary classification.
    Features: [Age, Income]
    Target: 1 if 'Buys Product', 0 otherwise.
    """
    np.random.seed(42)
    # 100 samples, 2 features (Age, Income)
    # Ages from 18 to 70, Income from 20k to 100k
    X = np.zeros((100, 2))
    X[:, 0] = np.random.randint(18, 70, size=100)
    X[:, 1] = np.random.randint(20, 100, size=100)
    
    # Logic: If Age > 30 and Income > 50, then 1, else 0 (with some noise)
    y = ((X[:, 0] > 30) & (X[:, 1] > 50)).astype(int)
    # Add some noise to make it more realistic
    noise = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    y = np.abs(y - noise)
    return X, y

def main():
    print("--- Decision Tree from Scratch ---")
    
    # 1. Generate Data
    X, y = generate_dataset()
    
    # 2. Split into Train/Test
    # Simple split for demonstration
    train_size = 80
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 3. Train Model
    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    
    # 4. Predict
    predictions = clf.predict(X_test)
    
    # 5. Accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Model trained on {len(X_train)} samples.")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # 6. Sample Predictions
    print("\nSample Predictions (First 5 test samples):")
    for i in range(5):
        print(f"Features (Age, Income): {X_test[i]}, Actual: {y_test[i]}, Predicted: {predictions[i]}")

if __name__ == "__main__":
    main()
