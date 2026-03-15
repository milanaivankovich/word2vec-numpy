import numpy as np
import random


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def negative_sampling(vocab_size, positive_id, K):
    negatives = []
    while len(negatives) < K:
        neg = random.randint(1, vocab_size - 1)
        if neg != positive_id:
            negatives.append(neg)
    return negatives


class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, lr=0.025):
        self.V = vocab_size
        self.D = embedding_dim
        self.initial_lr = lr
        self.lr = lr

        self.W_in = np.random.randn(self.V, self.D) * 0.01
        self.W_out = np.random.randn(self.V, self.D) * 0.01

    def forward(self, v_c, v_o, v_n):
        score_pos = sigmoid(np.dot(v_o, v_c))
        score_neg = sigmoid(np.dot(v_n, v_c))

        loss = -np.log(score_pos + 1e-10) - np.sum(np.log(1 - score_neg + 1e-10))

        return score_pos, score_neg, loss

    def backward(self, score_pos, score_neg, v_o, v_n):
        grad_pos = score_pos - 1
        grad_neg = score_neg

        grad_v_c = grad_pos * v_o + np.dot(grad_neg, v_n)

        return grad_pos, grad_neg, grad_v_c

    def update_weights(self, center, context, negatives, grad_pos, grad_neg, grad_v_c, v_c):
        self.W_out[context] -= self.lr * grad_pos * v_c
        self.W_out[negatives] -= self.lr * np.outer(grad_neg, v_c)
        self.W_in[center] -= self.lr * grad_v_c

    def train_step(self, center, context, negatives):
        v_c = self.W_in[center]
        v_o = self.W_out[context]
        v_n = self.W_out[negatives]

        score_pos, score_neg, loss = self.forward(v_c, v_o, v_n)

        grad_pos, grad_neg, grad_v_c = self.backward(score_pos, score_neg, v_o, v_n)

        self.update_weights(center, context, negatives, grad_pos, grad_neg, grad_v_c, v_c)

        return loss

    def train(self, pairs, epochs, K):
        history = []

        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(pairs)

            self.lr = self.initial_lr * (1 - epoch / epochs)
            if self.lr < 0.0001:
                self.lr = 0.0001

            for center, context in pairs:
                negatives = negative_sampling(self.V, context, K)
                loss = self.train_step(center, context, negatives)
                total_loss += loss

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, LR: {self.lr:.4f}")
            history.append(total_loss)

        return history

    def most_similar(self, word, w2i, i2w, top_k=5):
        if word not in w2i:
            return [f"Word '{word}' not in vocabulary."]

        vec = self.W_in[w2i[word]]

        sims = np.dot(self.W_in, vec)

        norms = np.linalg.norm(self.W_in, axis=1) * np.linalg.norm(vec)

        norms[norms == 0] = 1e-10

        sims = sims / norms

        best = np.argsort(-sims)[:top_k + 1]

        return [i2w[i] for i in best if i2w[i] != word and i2w[i] != "<UNK>"][:top_k]