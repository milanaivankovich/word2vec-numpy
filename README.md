# Word2Vec in Pure NumPy 🚀

An **from-scratch implementation of the Word2Vec model** (Skip-gram with Negative Sampling) using only **standard Python and NumPy**.

This project was built to demonstrate understanding of **neural network architectures**, **forward propagation**, and **manual gradient derivation** without relying on automatic differentiation frameworks like **PyTorch** or **TensorFlow**.

---

# 📐 Mathematical Foundation

The model optimizes the **Negative Sampling objective function**.

For a **center word** \(c\), a **positive context word** \(o\), and \(K\) **negative samples** \(n_k\), the loss is minimized as:

\[
E = -\log(\sigma(v_o^T v_c)) - \sum_{k=1}^{K} \log(1 - \sigma(v_{n_k}^T v_c))
\]

Gradients are **manually derived** and applied via **Stochastic Gradient Descent (SGD)** with **learning rate decay**.

---

# 🚀 Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/milanaivankovich/word2vec-numpy.git
cd word2vec-numpy
```

## 2. Install dependencies

**Note:** `scikit-learn` is used strictly to download the **20 Newsgroups (`sci.space`) text corpus**, not for any machine learning operations.

```bash
pip install -r requirements.txt
```

## 3. Run the training script

```bash
python train.py
```

---

# 📊 Sample Output

During training, the script outputs the **training loss** and the **decaying learning rate**.

After training, it evaluates the learned embeddings using **Cosine Similarity**.
![image alt](https://github.com/milanaivankovich/word2vec-numpy/blob/main/screenshot/sc_output.png)
![image alt](https://github.com/milanaivankovich/word2vec-numpy/blob/main/screenshot/sc%202.png)

---

# 🧠 Key Features

- **Pure Linear Algebra**  
  All forward passes, loss calculations (**Binary Cross-Entropy**), and weight updates via backpropagation are implemented manually using `numpy`.

- **Modular Architecture**  
  The training loop is cleanly separated into:

  - `forward()`
  - `backward()`
  - `update_weights()`

  This mirrors the design patterns used in modern ML frameworks.

- **Algorithmic Efficiency**  
  Uses **O(1) embedding lookups** by directly indexing weight matrices instead of computing expensive **one-hot vector matrix multiplications**.

- **Zero NLP Dependencies**  
  Text preprocessing, tokenization, and stop-word removal are implemented using **pure Python regular expressions**, avoiding external NLP libraries.

---

# 📂 Project Structure

```text
word2vec-numpy/
│
├── src/
│   ├── utils.py          # Text cleaning, vocabulary building, and pair generation
│   └── word2vec.py       # Skip-gram Negative Sampling model + manual backprop
│
├── requirements.txt      # Project dependencies
├── train.py              # Main training script
└── README.md             # Project documentation
```

---

# 🎯 Goal of the Project

The primary goal of this project is **educational** — to deeply understand how **Word2Vec and neural embeddings work internally**, without relying on high-level machine learning frameworks.

It demonstrates:

- manual neural network implementation
- gradient derivation
- negative sampling optimization
- efficient embedding lookup strategies

  ---

# ⚠️ Known Simplifications & Design Philosophy

To balance educational clarity with execution speed in pure Python, a few deliberate design choices and simplifications were made compared to the original Mikolov paper:

- **Uniform Negative Sampling:** Instead of using the standard unigram distribution raised to the $0.75$ power ($U(w)^{0.75}$) to dampen highly frequent words, this implementation uses **uniform sampling**. This avoids the overhead of calculating and sampling from probability distributions in pure Python.
- **No Frequent Word Subsampling:** The standard practice of probabilistically discarding highly frequent words (like "the", "and") during pair generation is omitted in favor of a strict, custom stop-word filter.
- **Small Vocabulary Experimentation:** The vocabulary is intentionally capped to ensure the pure-Python training loop runs reasonably fast on a standard CPU without needing Cython or C++ backends.
- **No Autograd:** The project strictly avoids automatic differentiation to demonstrate a transparent, ground-up understanding of matrix calculus and the low-level mechanics of optimization.
