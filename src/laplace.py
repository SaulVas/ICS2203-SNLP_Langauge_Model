"""
Implementation of the laplace (add 1) language model.
"""
import sys
from language_model_abc import LanguageModel

class LaplaceLM(LanguageModel):
    def _default_uni_value(self):
        return float(1 / sum(self.uni_count.values()) + len(self.uni_count))

    def _generate_unigram_probs(self):
        total_tokens = float(sum(self.uni_count.values()))
        for key in self.uni_count:
            self.uni_probabilities[key] = ((self.uni_count[key] + 1)
                                           / (total_tokens + len(self.uni_count)))

    def _generate_bigram_probs(self):
        for key in self.bi_count:
            words = tuple(key.split())
            self.bi_probabilities[words] = ((self.bi_count[key] + 1)
                                            / (self.uni_count[words[0]] + len(self.uni_count)))

    def _generate_trigram_probs(self):
        for key in self.tri_count:
            words = tuple(key.split())
            bi_gram_key = words[0] + " " + words[1]
            self.tri_probabilities[words] = ((self.tri_count[key] + 1)
                                             / (self.bi_count[bi_gram_key] + len(self.uni_count)))

    def _get_bigram_probability(self, bigram):
        return self.bi_probabilities.get(bigram,
                                         1 / (self.uni_count.get(bigram[0], 1)
                                              + len(self.uni_count)))

    def _get_trigram_probability(self, trigram):
        return self.tri_probabilities.get(trigram,
                                          1 / (self.bi_count.get(trigram[:2], 1)
                                               + len(self.uni_count)))

    def uni_sentence_probability(self, words):
        return max(super().uni_sentence_probability(words), sys.float_info.min)

    def bi_sentence_probability(self, words):
        return max(super().bi_sentence_probability(words), sys.float_info.min)

    def tri_sentence_probability(self, words):
        return max(super().tri_sentence_probability(words), sys.float_info.min)

    def sentence_probability(self, words):
        return max(super().sentence_probability(words), sys.float_info.min)
