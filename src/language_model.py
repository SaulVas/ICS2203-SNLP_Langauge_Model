import string
from collections import defaultdict
from abc import ABC, abstractmethod

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
            while context[-1] not in ["</s>", ""]:
                max_prob_found = 0
                word = ""
                for key, value in self.tri_probabilities.items():
                    if (key[0:2] == context) and value > max_prob_found:
                        max_prob_found = value
                        word = key[-1]

                words.append(word)
                context = (context[-1], word)
                print(context)

        print(" ".join(words))
