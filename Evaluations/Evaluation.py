from .bleu import *
from .rouge import *
from .ptbtokenizer import *
import math


# Utility functions and scorers for validation


def score_predictions_bleu_rouge(predictions):
    """
    A functions which calculates bleu and rouge scores
    :param predictions:
    :return: The scores
    """
    tokenizer = PTBTokenizer()
    gens = {}
    refs = {}
    for p, i in zip(predictions, range(len(predictions))):
        gens[str(i)] = [p['predicted']]
        refs[str(i)] = [p['gt']]
    gens = tokenizer.tokenize(gens)
    refs = tokenizer.tokenize(refs)
    results = dict()
    print('Calculating ROUGE...')
    rouge_scorer = Rouge()
    rouge_avg_score, rouge_all_scores = rouge_scorer.compute_score(refs, gens)
    results['ROUGE'] = (rouge_avg_score, rouge_all_scores)
    print('Calculating BLEU...')
    bleu_scorer = Bleu(4)
    bleu_avg_score, bleu_all_scores = bleu_scorer.compute_score(refs, gens)
    for n, bleu_n_avg_score, bleu_n_all_scores in zip(range(len(bleu_avg_score)), bleu_avg_score, bleu_all_scores):
        results[f'BLEU-{n + 1}'] = (bleu_n_avg_score, bleu_n_all_scores)

    return results


def perplexity(model, test_data):
    """
    :param test_data: (list of str), sentences comprising the test corpus
    :return: The perplexity of the model as a float.
    """
    test_tokens = preprocess(test_data, self.n)
    test_ngrams = nltk.ngrams(test_tokens, self.n)
    N = len(test_tokens)

    known_ngrams = (self._convert_oov(ngram) for ngram in test_ngrams)
    probabilities = [model[ngram] for ngram in known_ngrams]

    return math.exp((-1 / N) * sum(map(math.log, probabilities)))


if __name__ == "__main__":
    pass
    # Get the predictions
    # Get the bleu and rouge scores
