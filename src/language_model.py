import string
from collections import defaultdict
from abc import ABC, abstractmethod
import random

class LanguageModel(ABC):
    def __init__(self):
        self.uni_count = defaultdict(int)
        self.bi_count = defaultdict(int)
        self.tri_count = defaultdict(int)
        self.uni_probabilities = defaultdict(int)
        self.bi_probabilities = defaultdict(int)
        self.tri_probabilities = defaultdict(int)

        self._get_counts()
        self._uni_gram_prob()
        self._bi_gram_prob()
        self._tri_gram_prob()

    def __str__(self):
        ret_str =  (f"uni_count has {len(self.uni_count.keys())} tokens\n"
                + f"bi_count has {len(self.bi_count.keys())} tokens\n"
                + f"tri_count has {len(self.tri_count.keys())} tokens\n")
        return ret_str

    @abstractmethod
    def _get_counts(self):
        pass

    @abstractmethod
    def _uni_gram_prob(self):
        pass

    @abstractmethod
    def _bi_gram_prob(self):
        pass

    @abstractmethod
    def _tri_gram_prob(self):
        pass

    def _remove_punctuation(self, text):
        """
        Removes all punctuation from the given text, except for the single quote.

        Args:
            text (str): The input text.

        Returns:
            str: The text with punctuation removed.
        """
        punctuation = string.punctuation.replace("'", "")
        text_without_punctuation = text.translate(str.maketrans("", "", punctuation))

        return text_without_punctuation

    def text_generator(self, phrase):
        phrase = self._remove_punctuation(phrase)
        words = phrase.lower().split()  # Convert to lowercase and split into words
        words.insert(0, "<s>")
        if len(words) > 1:
            context = tuple(words[-2:])
            loop_prevention_counter = 0
            while context[-1] not in ["</s>", ""] and loop_prevention_counter < 100:
                token_probabilities = {}

                for key, value in self.tri_probabilities.items():
                    if key[0:2] == context:
                        token_probabilities[key[-1]] = value

                if not token_probabilities:
                    break

                # semi-random selection of next word
                random_dec = random.random()
                probabilities_sum = 0
                for token, probability in sorted(token_probabilities.items(),
                                                 key=lambda item: item[1]):
                    probabilities_sum += probability
                    if probabilities_sum > random_dec:
                        word = token
                        break

                words.append(word)
                context = (context[-1], word)
                loop_prevention_counter += 1

        if words[-1] != "</s>":
            words.append("</s>")

        print(" ".join(words))
