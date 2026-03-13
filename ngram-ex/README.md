# N-Gram Language Model

A simple n-gram language model that builds from a text corpus, computes conditional probabilities with Laplace smoothing, and generates new sentences.

## Features

- Build unigram, bigram, or higher-order n-gram models from a corpus file
- Compute per-n-gram conditional probabilities with add-one (Laplace) smoothing
- Generate sentences via weighted random sampling

## Requirements

Python 3.11.

## Corpus

By default, the script expects a corpus file at `assets/corpus-100mb.txt` — a plain text file with one sentence per line. Override this with `--corpus`.

## Usage

```bash
python n-gram.py [--n N] [--sentence SENTENCE] [--num-sentences NUM] [--corpus PATH]
```

### Arguments

| Argument            | Default                         | Description                                         |
| ------------------- | ------------------------------- | --------------------------------------------------- |
| `--n`             | `1`                           | Order of the n-gram (1 = unigram, 2 = bigram, etc.) |
| `--sentence`      | `"Hôm nay trời đẹp lắm"` | Input sentence to compute n-gram probabilities for  |
| `--num-sentences` | `5`                           | Number of sentences to generate from the model      |
| `--corpus`        | `assets/corpus-100mb.txt`     | Path to the corpus file                             |

### Examples

Build a bigram model and compute probabilities for a custom sentence:

```bash
python n-gram.py --n 2 --sentence "Hôm nay trời đẹp lắm"
```

Use a custom corpus and generate 3 sentences:

```bash
python n-gram.py --n 2 --corpus path/to/corpus.txt --num-sentences 3
```

## How It Works

1. **Preprocessing** — punctuation is stripped and text is lowercased. Each sentence is wrapped with `<s>` and `</s>` boundary tokens.
2. **Model building** — the corpus is read line by line, counting all n-grams and unigrams to build the vocabulary.
3. **Probability estimation** — for a given input sentence, each n-gram's conditional probability is computed using Laplace smoothing:

   ```
   P(w_i | w_{i-1}) = (C(w_{i-1}, w_i) + 1) / (C(w_{i-1}) + |V|)
   ```
4. **Sentence generation** — starting from `<s>`, the next token is sampled from all n-grams sharing the current prefix, weighted by their corpus counts, until `</s>` is reached or `max_length` is hit.
