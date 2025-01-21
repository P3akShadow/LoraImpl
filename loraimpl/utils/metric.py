import math
from collections import Counter

import nltk
import numpy as np


class CIDEr:
    """
    Custom CIDEr implementation
    """

    @staticmethod
    def preprocess_text(text):
        return nltk.word_tokenize(text.lower().strip())

    @staticmethod
    def compute_ngrams(words, n):
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

    @classmethod
    def compute_tf(cls, words, n):
        ngrams = cls.compute_ngrams(words, n)
        counter = Counter(ngrams)
        total = sum(counter.values())
        return {gram: count/total for gram, count in counter.items()} if total > 0 else counter

    @classmethod
    def compute_idf(cls, all_refs, n):
        doc_count = Counter()
        total_docs = len(all_refs)

        for refs in all_refs:
            seen_grams = set()
            for ref in refs:
                words = cls.preprocess_text(ref)
                ngrams = cls.compute_ngrams(words, n)
                seen_grams.update(ngrams)
            doc_count.update(seen_grams)

        idf_dict = {gram: math.log(total_docs/(count + 1)) for gram, count in doc_count.items()}
        return idf_dict

    @classmethod
    def compute_cider_score(cls, pred, refs, n, idf):
        pred_words = cls.preprocess_text(pred)
        pred_tf = cls.compute_tf(pred_words, n)

        scores = []
        for ref in refs:
            ref_words = cls.preprocess_text(ref)
            ref_tf = cls.compute_tf(ref_words, n)

            common_grams = set(pred_tf.keys()) & set(ref_tf.keys())

            if len(common_grams) == 0:
                continue

            numerator = sum(pred_tf[gram] * ref_tf[gram] * (idf.get(gram, 0) ** 2) for gram in common_grams)

            pred_norm = math.sqrt(sum((tf * idf.get(gram, 0) ** 2) ** 2 for gram, tf in pred_tf.items()))
            ref_norm = math.sqrt(sum((tf * idf.get(gram, 0) ** 2) ** 2 for gram, tf in ref_tf.items()))

            if pred_norm > 0 and ref_norm > 0:
                score = numerator / (pred_norm * ref_norm)
                scores.append(score)

        return max(scores) if scores else 0

    @classmethod
    def compute(cls, predictions, references):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # take only first reference. #TODO: @Morgan, if you've got the time, please fix the implementation for multiple references
        references = [[ref[0]] for ref in references]
        if not predictions or not references or len(predictions) != len(references):
            return 0.0

        n_values = range(1, 5)
        weights = [1/4] * 4

        scores = []
        for n in n_values:
            idf = cls.compute_idf(references, n)
            score = np.mean([cls.compute_cider_score(pred, refs, n, idf)
                             for pred, refs in zip(predictions, references)])
            scores.append(score)

        return {'cider': sum(w * s for w, s in zip(weights, scores))}

