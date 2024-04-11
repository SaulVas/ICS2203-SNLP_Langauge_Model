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
    
