import numpy as np


class BigramLanguageModel:
    """
    A simple Bigram Language Model implemented from scratch using NumPy.
    It learns the probability of a word given the previous word.
    """

    def __init__(self):
        self.vocab = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.probabilities = None

    def tokenize(self, text):
        """Simple whitespace tokenization."""
        return text.lower().split()

    def build_vocab(self, tokens):
        """Builds the vocabulary and mapping dictionaries."""
        self.vocab = sorted(list(set(tokens)))
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}
        return len(self.vocab)

    def train(self, tokens, smoothing=0.1):
        """
        Trains the model by counting bigrams and normalizing to probabilities.
        Uses Laplace-style smoothing to avoid zero probabilities.
        """
        vocab_size = self.build_vocab(tokens)
        counts = np.zeros((vocab_size, vocab_size))

        # Count occurrences of each bigram
        for i in range(len(tokens) - 1):
            curr_idx = self.word_to_idx[tokens[i]]
            next_idx = self.word_to_idx[tokens[i + 1]]
            counts[curr_idx, next_idx] += 1

        # Apply smoothing
        counts += smoothing

        # Normalize to get probabilities (each row sums to 1)
        self.probabilities = counts / counts.sum(axis=1, keepdims=True)
        print(
            f"Model trained on {len(tokens)} tokens with a vocabulary of {vocab_size} words."
        )

    def generate(self, start_word, length=10, temperature=1.0):
        """Generates a sequence of words and calculates the average probability (confidence).

        Args:
            start_word: The seed word to start generation
            length: Number of words to generate
            temperature: Controls randomness (lower = more confident/deterministic, higher = more random)
                        Default 1.0 = normal sampling, 0.5 = more confident, 0.1 = very confident
        """
        if start_word.lower() not in self.word_to_idx:
            return f"Error: '{start_word}' not in vocabulary.", 0

        current_word = start_word.lower()
        result = [current_word]
        total_log_prob = 0

        for _ in range(length - 1):
            curr_idx = self.word_to_idx[current_word]
            probs = self.probabilities[curr_idx]

            # Apply temperature to probabilities
            if temperature != 1.0:
                # Adjust probabilities with temperature (lower temp = sharper distribution)
                log_probs = np.log(probs + 1e-10)  # Add small constant to avoid log(0)
                log_probs = log_probs / temperature
                # Renormalize
                exp_probs = np.exp(
                    log_probs - np.max(log_probs)
                )  # Subtract max for numerical stability
                probs = exp_probs / exp_probs.sum()

            # Sample the next word index
            next_idx = np.random.choice(len(self.vocab), p=probs)
            next_word = self.idx_to_word[next_idx]

            # Track the probability of the chosen word for "accuracy" metric
            total_log_prob += np.log(probs[next_idx])

            result.append(next_word)
            current_word = next_word

        # Perplexity is a common metric for LM quality: exp(-average log probability)
        # Lower perplexity = better model. We'll also provide a "Confidence" score.
        avg_log_prob = total_log_prob / (length - 1)
        confidence = np.exp(avg_log_prob)

        return " ".join(result), confidence


if __name__ == "__main__":
    # Quick test
    text = "the quick brown fox jumps over the lazy dog"
    model = BigramLanguageModel()
    tokens = model.tokenize(text)
    model.train(tokens)
    print("Generated:", model.generate("the", length=5))
