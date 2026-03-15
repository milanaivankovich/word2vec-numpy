import re
from collections import Counter
from sklearn.datasets import fetch_20newsgroups


def clean_text_pure_python(text):

    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)


    stop_words = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'is', 'it', 'of', 'for', 'with', 'that',
                  'this', 'from', 'by', 'as', 'are', 'was'}


    return [t for t in tokens if t not in stop_words and len(t) > 2]


def load_and_preprocess_data(max_vocab_size=2000, window_size=2):
    print("1. Fetching 'sci.space' dataset from 20newsgroups...")
    newsgroups = fetch_20newsgroups(subset='train', categories=['sci.space'], remove=('headers', 'footers', 'quotes'))

    all_tokens = []
    processed_docs = []

    print("2. Cleaning text (Pure Python)...")
    for doc in newsgroups.data:
        clean_tokens = clean_text_pure_python(doc)
        if len(clean_tokens) > 1:
            processed_docs.append(clean_tokens)
            all_tokens.extend(clean_tokens)

    print("3. Building Vocabulary...")
    word_counts = Counter(all_tokens)
    top_words = word_counts.most_common(max_vocab_size - 1)

    vocab = ["<UNK>"] + [w for w, count in top_words]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}

    print("4. Generating context pairs...")
    pairs = []
    for doc in processed_docs:
        encoded_doc = [word_to_id.get(w, 0) for w in doc]
        encoded_doc = [w for w in encoded_doc if w != 0]

        for i, center in enumerate(encoded_doc):
            for j in range(max(0, i - window_size), min(len(encoded_doc), i + window_size + 1)):
                if i != j:
                    pairs.append((center, encoded_doc[j]))

    return pairs, word_to_id, id_to_word, len(vocab)