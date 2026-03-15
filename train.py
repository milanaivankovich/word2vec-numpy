import matplotlib.pyplot as plt
from src.utils import load_and_preprocess_data
from src.word2vec import Word2Vec


def main():
    vocab_limit = 1500
    pairs, w2i, i2w, V = load_and_preprocess_data(max_vocab_size=vocab_limit, window_size=2)

    print(f"\nStarting training with {len(pairs)} context pairs...")
    model = Word2Vec(vocab_size=V, embedding_dim=50, lr=0.05)

    epochs = 5
    loss_history = model.train(pairs, epochs=epochs, K=5)

    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)


    plt.savefig("sc.png")
    plt.show()

    print("\n--- TESTING SEMANTIC SIMILARITY ---")
    test_words = ["orbit", "moon", "launch", "earth"]

    for word in test_words:
        similar_words = model.most_similar(word, w2i, i2w)
        print(f"Most similar to '{word}': {similar_words}")


if __name__ == "__main__":
    main()