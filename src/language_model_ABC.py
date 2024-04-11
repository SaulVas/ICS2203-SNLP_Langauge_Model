"""
Implements an abstract base class for language models
"""
import string
import random
from collections import defaultdict
from abc import ABC, abstractmethod

class LanguageModel(ABC):
    """
    Abstract base class for language models.

    This class defines the common structure and methods for language models.
    Subclasses must implement the abstract methods to provide specific functionality.

    Attributes:
        uni_count (defaultdict): A dictionary to store the counts of unigrams.
        bi_count (defaultdict): A dictionary to store the counts of bigrams.
        tri_count (defaultdict): A dictionary to store the counts of trigrams.
        uni_probabilities (defaultdict): A dictionary to store the probabilities of unigrams.
        bi_probabilities (defaultdict): A dictionary to store the probabilities of bigrams.
        tri_probabilities (defaultdict): A dictionary to store the probabilities of trigrams.

    Methods:
        __init__(): Initializes the language model and calculates the counts and probabilities.
        __str__(): Returns a string representation of the language model.
        _get_counts(): Loads or generates the n-gram counts.
        _uni_gram_prob(): Calculates the unigram probabilities.
        _bi_gram_prob(): Calculates the bigram probabilities.
        _tri_gram_prob(): Calculates the trigram probabilities.
        _remove_punctuation(text): Removes punctuation from the given text.
        text_generator(phrase): Generates text based on a given phrase using the language model.
    """

    def __init__(self):
        """
        Initializes the language model and calculates the counts and probabilities.
        """
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
        """
        Returns a string representation of the language model.

        Returns:
            str: A string representation of the language model.
        """
        ret_str =  (f"uni_count has {len(self.uni_count.keys())} tokens\n"
                + f"bi_count has {len(self.bi_count.keys())} tokens\n"
                + f"tri_count has {len(self.tri_count.keys())} tokens\n")
        return ret_str

    @abstractmethod
    def _get_counts(self):
        """
        Loads the n-gram counts from JSON files if they exist, otherwise generates the counts.

        If the JSON files for 1-gram, 2-gram, and 3-gram counts exist in the 'n_grams/vanilla'
        directory, this method loads the counts from the files and assigns them to the 
        corresponding instance variables. If the files do not exist, it calls the 
        '_generate_counts' method to generate the counts.
        """

    @abstractmethod
    def _uni_gram_prob(self):
        """
        Calculates unigram probabilities.

        The function calculates the probability of each unigram based on the total token count.
        """

    @abstractmethod
    def _bi_gram_prob(self):
        """
        Calculates bigram probabilities.

        The function calculates the probability of each bigram based on the bigram counts and
        corresponding unigram counts.
        """

    @abstractmethod
    def _tri_gram_prob(self):
        """
        Calculates trigram probabilities.

        The function calculates the probability of each trigram based on the trigram counts and
        corresponding bigram counts.
        """

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
        """
        Generates text based on a given phrase using a language model.

        Args:
            phrase (str): The input phrase to generate text from.
        """
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

    @abstractmethod
    def sentence_probability(self, sentence):
        """
        Calculate the probability of a given sentence according to the language model.

        Parameters:
            sentence (str): The input sentence for which the probability needs to be calculated.

        Returns:
            float: The probability of the given sentence according to the language model.
        """
