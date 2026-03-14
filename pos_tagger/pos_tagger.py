"""
POS Tagging – Brown Corpus (OOD Refactor)
IT4772 – Natural Language Processing
"""

import nltk
from nltk.corpus import brown, treebank
from nltk.tag import UnigramTagger, DefaultTagger, map_tag
from sklearn.metrics import precision_recall_fscore_support, classification_report
from abc import ABC, abstractmethod


class ResourceLoader:
    """Handles downloading and loading of NLTK resources."""
    
    @staticmethod
    def ensure_resources():
        resources = [
            "brown", "treebank", "universal_tagset",
            "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"
        ]
        for res in resources:
            nltk.download(res, quiet=True)


class CorpusData:
    """Encapsulates corpus sentences and their tags mapped to the Universal tagset."""
    
    def __init__(self, tagged_sents, tagset_from="en-ptb"):
        # Map sentences to universal tagset
        self.sents_universal = [
            [(w, map_tag(tagset_from, "universal", t)) for w, t in sent]
            for sent in tagged_sents
        ]
        
        # Extract features
        self.words_per_sent = [[w for w, _ in sent] for sent in self.sents_universal]
        self.true_tags = [tag for sent in self.sents_universal for _, tag in sent]
        self.labels = sorted(set(self.true_tags))

    @property
    def total_sents(self):
        return len(self.sents_universal)

    @property
    def total_tokens(self):
        return len(self.true_tags)


class BaseTagger(ABC):
    """Abstract base class/Interface for all POS taggers."""
    
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def tag_sentences(self, sents_words):
        """Tags a list of tokenized sentences. Returns a flat list of universal tags."""
        pass


class PerceptronTaggerX(BaseTagger):
    """Tagger X: Pre-trained NLTK PerceptronTagger."""
    
    def __init__(self):
        super().__init__("PerceptronTagger (X)")

    def tag_sentences(self, sents_words):
        pred_tags = []
        for sent_words in sents_words:
            # nltk.pos_tag uses en-ptb tagset natively
            tagged = nltk.pos_tag(sent_words)
            pred_tags.extend(map_tag("en-ptb", "universal", t) for _, t in tagged)
        return pred_tags


class UnigramTreebankTaggerY(BaseTagger):
    """Tagger Y: UnigramTagger trained on the Penn Treebank."""
    
    def __init__(self, treebank_corpus: CorpusData):
        super().__init__("UnigramTagger/TB (Y)")
        default_tagger = DefaultTagger("NOUN")
        self.tagger = UnigramTagger(treebank_corpus.sents_universal, backoff=default_tagger)

    def tag_sentences(self, sents_words):
        pred_tags = []
        for sent_words in sents_words:
            tagged = self.tagger.tag(sent_words)
            pred_tags.extend(t for _, t in tagged)
        return pred_tags


class Evaluator:
    """Evaluates tagger predictions against golden true tags."""
    
    def __init__(self, true_tags, labels):
        self.true_tags = true_tags
        self.labels = labels
        self.results = []

    def evaluate(self, tagger: BaseTagger, pred_tags):
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            self.true_tags, pred_tags, average="macro", labels=self.labels, zero_division=0
        )
        report = classification_report(
            self.true_tags, pred_tags, labels=self.labels, zero_division=0, digits=4
        )
        
        result = {
            "name": tagger.name,
            "macro_p": macro_p,
            "macro_r": macro_r,
            "macro_f1": macro_f1,
            "report": report
        }
        self.results.append(result)
        return result

    def print_summary(self):
        print("\n" + "=" * 62)
        print(f"{'Tagger':<26} {'Macro-P':>10} {'Macro-R':>10} {'Macro-F1':>10}")
        print("─" * 62)
        for r in self.results:
            print(f"{r['name']:<26} {r['macro_p']:>10.4f} {r['macro_r']:>10.4f} {r['macro_f1']:>10.4f}")
        print("=" * 62)

        for r in self.results:
            print(f"\n--- Per-class: {r['name']} ---")
            print(r["report"])


def main():
    # 1. Download/Install Resources
    ResourceLoader.ensure_resources()

    # 2. Data Preparation
    print("Loading corpora...")
    brown_data = CorpusData(brown.tagged_sents())
    print(f"Brown corpus (Golden): {brown_data.total_sents:,} sentences | {brown_data.total_tokens:,} tokens")
    
    treebank_data = CorpusData(treebank.tagged_sents())
    print(f"Treebank corpus (Train): {treebank_data.total_sents:,} sentences | {treebank_data.total_tokens:,} tokens")

    # 3. Tagger Initialization
    print("\nInitializing taggers...")
    tagger_x = PerceptronTaggerX()
    tagger_y = UnigramTreebankTaggerY(treebank_data)

    # 4. Predictions
    print("\nPredicting with Tagger X...")
    preds_x = tagger_x.tag_sentences(brown_data.words_per_sent)
    print(f"  Tagged {len(preds_x):,} tokens.")

    print("\nPredicting with Tagger Y...")
    preds_y = tagger_y.tag_sentences(brown_data.words_per_sent)
    print(f"  Tagged {len(preds_y):,} tokens.")

    # 5. Evaluation & Output
    evaluator = Evaluator(brown_data.true_tags, brown_data.labels)
    evaluator.evaluate(tagger_x, preds_x)
    evaluator.evaluate(tagger_y, preds_y)
    
    evaluator.print_summary()


if __name__ == "__main__":
    main()
