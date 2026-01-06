from bigram_lm import BigramLanguageModel


def main():
    # 1. Load the dataset
    try:
        with open("corpus.txt", "r") as f:
            text = f.read()
    except FileNotFoundError:
        print("Error: corpus.txt not found.")
        return

    # 2. Initialize and Tokenize
    model = BigramLanguageModel()
    tokens = model.tokenize(text)

    # 3. Train the model
    print("--- Training Mini Language Model ---")
    model.train(tokens, smoothing=0.0001)  # Minimal smoothing for higher confidence

    # 4. Generate some text with different temperature settings
    print("\n--- Generating Text (Temperature=0.2 for higher confidence) ---")
    seeds = ["the", "a", "fox", "dog"]
    for seed in seeds:
        generated, confidence = model.generate(seed, length=8, temperature=0.2)
        print(f"Seed '{seed}': {generated}")
        print(f"Confidence (Avg Prob): {confidence:.2%}\n")


if __name__ == "__main__":
    main()
