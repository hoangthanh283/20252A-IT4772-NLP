import argparse
import re
import random
from collections import Counter
from typing import Dict, Tuple, List


class NGramModel:
    def __init__(self, n: int, corpus_file: str):
        self.n = n
        self.ngram_counts = Counter()
        self.total_ngrams = 0
        self.corpus_file = corpus_file
        self.unigram_counts: Counter = Counter()  # C(w_{i-1}) for smoothing.
        self.vocab: set = set()
        self.ngram_model = self.build_ngram_model(n=self.n)

    def preprocess_sentence(self, sentence):
        """
        Preprocess the input sentence by removing punctuation, converting to lowercase, and splitting into words.

        Args:
            sentence: The input sentence to preprocess.

        Returns:
            A list of preprocessed words from the input sentence.
        """
        # Remove punctuation and convert to lowercase.
        sentence = re.sub(r'[^\w\s]', "", sentence.lower())
        # Remove spaces.
        words = sentence.split()
        words = [ww for ww in words if ww.strip()]
        return words

    def create_ngrams(self, text: str, n: int):
        """
        Create n-grams from the given text.

        Parameters:
        text (str): The input text to create n-grams from.
        n (int): The number of words in each n-gram.

        Returns:
        list: A list of n-grams.
        """
        words = self.preprocess_sentence(text)
        tokens = ["<s>"] + words + ["</s>"]
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        return ngrams

    def build_ngram_model(self, n: int = 1) -> Counter:
        """
        Build an n-gram model from a local corpus file.

        Args:
            n: The number of words in each n-gram.

        Returns:
            An n-gram model represented as a Counter object containing the frequency of each n-gram in the corpus.
        """
        ngram_counts = Counter()
        line_count = 0
        print(f"Processing {self.corpus_file}...")
        with open(self.corpus_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = self.preprocess_sentence(line)
                if not words:
                    continue
                tokens = ["<s>"] + words + ["</s>"]
                # Count unigrams and build vocab
                for token in tokens:
                    self.unigram_counts[token] += 1
                    self.vocab.add(token)
                # Count n-grams
                ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                ngram_counts.update(ngrams)
                line_count += 1
                if line_count % 500000 == 0:
                    print(f"  Processed {line_count} lines...")
        print(f"Finished processing. Total lines: {line_count}.")
        print(f"Vocabulary size: {len(self.vocab)}")
        return ngram_counts

    def compute_ngram_probs_for_sentence(self, sentence: str, n: int) -> Dict[tuple, float]:
        """
        Compute the probabilities of n-grams in a given sentence based on the provided n-gram model.
        Uses add-one (Laplace) smoothing to avoid zero probabilities.

        Args:
            sentence: The input sentence for which to compute n-gram probabilities.
            n: The number of words in each n-gram.

        Returns:
            A dictionary where keys are n-grams (as tuples) and values are their corresponding probabilities.
        """
        sentence_ngrams = self.create_ngrams(sentence, n)
        vocab_size = len(self.vocab)

        ngram_probs: Dict[Tuple[str, str], float] = {}
        for ngram in sentence_ngrams:
            bigram_count = self.ngram_model.get(ngram, 0)
            # For bigram (w_{i-1}, w_i): condition on w_{i-1}
            prev_word = ngram[0]
            unigram_count = self.unigram_counts.get(prev_word, 0)
            # Laplace smoothing: P(w_i | w_{i-1}) = (C(w_{i-1}, w_i) + 1) / (C(w_{i-1}) + |V|)
            prob = (bigram_count + 1) / (unigram_count + vocab_size) if (unigram_count + vocab_size) > 0 else 0
            ngram_probs[ngram] = prob
            print(f"  P({ngram[1]} | {ngram[0]}) = {prob:.6f}  "
                  f"[C({ngram[0]}, {ngram[1]}) = {bigram_count}, C({ngram[0]}) = {unigram_count}]")
        return ngram_probs
    
    def generate_sentence(self, max_length: int = 30) -> str:
        """
        Generate a random sentence based on the n-gram model using weighted sampling.

        Args:
            max_length: The maximum length of the generated sentence.

        Returns:
            A generated sentence as a string.
        """
        if not self.ngram_model:
            return ""

        sentence: List[str] = []
        current_token = "<s>"
        for _ in range(max_length):
            # Get all n-grams that start with the current token.
            candidates = [(ngram, count) for ngram, count in self.ngram_model.items() if ngram[0] == current_token]
            if not candidates:
                break
            
            # Separate n-grams and their counts.
            ngrams, counts = zip(*candidates)
            total_counts = sum(counts)
            probabilities = [count / total_counts for count in counts]

            # Randomly select the next n-gram based on the computed probabilities.
            next_ngram = random.choices(ngrams, weights=probabilities, k=1)[0]
            next_token = next_ngram[1]
            if next_token == "</s>":
                break
            sentence.append(next_token)
            current_token = next_token
        return " ".join(sentence)


def main(parser):
    args = parser.parse_args()
    ngram_model = NGramModel(n=args.n, corpus_file=args.corpus)

    # Calculate and print n-gram probabilities for the input sentence.
    print(f"Built {args.n}-gram model with {len(ngram_model.ngram_model)} unique n-grams.")
    bigram_probs = ngram_model.compute_ngram_probs_for_sentence(args.sentence, args.n)
    for bigram, prob in bigram_probs.items():
        print(f"{bigram}: {prob:.6f}")
    
    # Generate and print a new sentence based on the n-gram model.
    for _ in range(args.num_sentences):
        generated_sentence = ngram_model.generate_sentence()
        print(f"Generated sentence: {generated_sentence}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1, help="The number of words in each n-gram")
    parser.add_argument("--sentence", type=str, default="Hôm nay trời đẹp lắm", help="The sentence to compute n-gram probabilities for")
    parser.add_argument("--num-sentences", type=int, default=5, help="The number of sentences to generate")
    parser.add_argument("--corpus", type=str, default="assets/corpus-100mb.txt", help="Path to the corpus file")
    main(parser)
