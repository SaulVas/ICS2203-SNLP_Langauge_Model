"""
Implementation of the Vanilla language model class
"""
from language_model_abc import LanguageModel

class VanillaLM(LanguageModel):
    """
    Language model implementation using vanilla n-gram approach.

    This class inherits from the LanguageModel abstract base class and provides an implementation
    for the _get_counts, _generate_unigram_probs, _generate_bigram_probs, and _generate_trigram_probs methods.
    """
    def _default_uni_value(self):
        return 0.0

    def _generate_unigram_probs(self):
        """
        Calculates unigram probabilities.

        The function calculates the probability of each unigram based on the total token count.

        Args:
            self: The instance of the language model.

        Returns:
            None
        """
        total_tokens = float(sum(self.uni_count.values()))
        for key in self.uni_count:
            self.uni_probabilities[key] = self.uni_count[key] / total_tokens

    def _generate_bigram_probs(self):
        """
        Calculates bigram probabilities.

        The function calculates the probability of each bigram based on the bigram counts and
        corresponding unigram counts.

        Args:
            self: The instance of the language model.

        Returns:
            None
        """
        for key in self.bi_count:
            words = tuple(key.split())
            self.bi_probabilities[words] = self.bi_count[key] / self.uni_count[words[0]]

    def _generate_trigram_probs(self):
        """
        Calculates trigram probabilities.

        The function calculates the probability of each trigram based on the trigram counts and
        corresponding bigram counts.

        Args:
            self: The instance of the language model.

        Returns:
            None
        """
        for key in self.tri_count:
            words = tuple(key.split())
            bi_gram_key = words[0] + " " + words[1]
            self.tri_probabilities[words] = self.tri_count[key] / self.bi_count[bi_gram_key]

    def _get_bigram_probability(self, bigram):
        return self.bi_probabilities[bigram]

    def _get_trigram_probability(self, trigram):
        return self.tri_probabilities[trigram]
