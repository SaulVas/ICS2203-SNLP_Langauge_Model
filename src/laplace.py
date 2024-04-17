"""
Implementation of the laplace (add 1) language model.
"""
from language_model_abc import LanguageModel

class LaplaceLM(LanguageModel):
    def _defualt_uni_value(self):
        return float(1 / sum(self.uni_count.values()) + len(self.uni_count))

    def _uni_gram_prob(self):
        total_tokens = float(sum(self.uni_count.values()))
        for key in self.uni_count:
            self.uni_probabilities[key] = ((self.uni_count[key] + 1)
                                           / (total_tokens + len(self.uni_count)))

    def _bi_gram_prob(self):
        for key in self.bi_count:
            words = tuple(key.split())
            self.bi_probabilities[words] = ((self.bi_count[words] + 1)
                                            / (self.uni_count[words[0]] + len(self.uni_count)))

    def _tri_gram_prob(self):
        for key in self.tri_count:
            words = tuple(key.split())
            bi_gram_key = words[0] + " " + words[1]
            self.tri_probabilities[words] = ((self.tri_count[words] + 1)
                                             / (self.bi_count[bi_gram_key] + len(self.uni_count)))

    def _linear_interpolation(self, trigram):
        uni_prob = 0.1 * self.uni_probabilities[trigram[-1]]

        bi_prob = 0.3 * self.bi_probabilities.get(trigram[-2:],
                                                  1
                                                  / (self.uni_count.get(trigram[0], 1)
                                                     + len(self.uni_count)))

        tri_prob = 0.6 * self.tri_probabilities.get(trigram,
                                                    1
                                                    / (self.bi_count.get(trigram[:2], 1)
                                                       + len(self.uni_count)))

        return uni_prob + bi_prob + tri_prob
