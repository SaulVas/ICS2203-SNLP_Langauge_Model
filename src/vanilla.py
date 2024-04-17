"""
Implementation of the Vanilla language model class
"""
from language_model_abc import LanguageModel

class VanillaLM(LanguageModel):
    """
    Language model implementation using vanilla n-gram approach.

    This class inherits from the LanguageModel abstract base class and provides an implementation
    for the _get_counts, _uni_gram_prob, _bi_gram_prob, and _tri_gram_prob methods.
    """
    def _defualt_uni_value(self):
        return float

    def _uni_gram_prob(self):
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

    def _bi_gram_prob(self):
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

    def _tri_gram_prob(self):
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

    def _linear_interpolation(self, trigram):
        return ((0.6 * self.tri_probabilities[trigram])
                + (0.3 * self.bi_probabilities[trigram[: 2]])
                + (0.1 * self.uni_probabilities[trigram[0]])
)
vanilla = VanillaLM()
vanilla.sentence_probability("siegler is a fucking faggot")
