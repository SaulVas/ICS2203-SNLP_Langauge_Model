"""
Implements an abstract base class for language models
"""
import string
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import os
from abc import ABC, abstractmethod
from dataset_functions import handle_sentence

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
        _generate_unigram_probs(): Calculates the unigram probabilities.
        _generate_bigram_probs(): Calculates the bigram probabilities.
        _generate_trigram_probs(): Calculates the trigram probabilities.
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
        self.uni_probabilities = defaultdict(self._default_uni_value)
        self.bi_probabilities = defaultdict(float)
        self.tri_probabilities = defaultdict(float)

        self._get_counts()
        self._generate_unigram_probs()
        self._generate_bigram_probs()
        self._generate_trigram_probs()

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
    def _default_uni_value(self):
        """"""

    def _get_counts(self):
        """
        Loads the n-gram counts from JSON files if they exist, otherwise generates the counts.

        If the JSON files for 1-gram, 2-gram, and 3-gram counts exist in the 'n_grams/vanilla_laplace'
        directory, this method loads the counts from the files and assigns them to the
        corresponding instance variables. If the files do not exist, it calls the
        '_generate_counts' method to generate the counts.

        Args:
            None

        Returns:
            None
        """
        if not (os.path.exists('n_grams/vanilla_laplace/1_gram_counts.json')
                and os.path.exists('n_grams/vanilla_laplace/2_gram_counts.json')
                and os.path.exists('n_grams/vanilla_laplace/3_gram_counts.json')):
            self._generate_counts()

        with open("n_grams/vanilla_laplace/1_gram_counts.json", 'r', encoding='utf-8') as fp:
            self.uni_count = json.load(fp)
        with open("n_grams/vanilla_laplace/2_gram_counts.json", 'r', encoding='utf-8') as fp:
            self.bi_count = json.load(fp)
        with open("n_grams/vanilla_laplace/3_gram_counts.json", 'r', encoding='utf-8') as fp:
            self.tri_count = json.load(fp)

    def _generate_counts(self):
        """
        Generate n-gram counts and save them to JSON files.

        This function iterates over a range of word counts (1 to 3) and generates n-gram counts
        based on the sentences in the training_set.xml file. The n-gram counts are then saved
        to separate JSON files for each word count.

        Args:
            self: The instance of the language model.

        Returns:
            None
        """
        for number_of_words in range(1, 4):
            n_gram_counts = defaultdict(int)
            tree = ET.parse('../data/training_set.xml')
            root = tree.getroot()
            for child in root:
                handle_sentence(child, number_of_words, n_gram_counts)

            with open(f'n_grams/vanilla_laplace/{number_of_words}_gram_counts.json',
                    'w', encoding='utf-8') as fp:
                json.dump(n_gram_counts, fp, indent=4)

    @abstractmethod
    def _generate_unigram_probs(self):
        """
        Calculates unigram probabilities.

        The function calculates the probability of each unigram based on the total token count.
        """

    @abstractmethod
    def _generate_bigram_probs(self):
        """
        Calculates bigram probabilities.

        The function calculates the probability of each bigram based on the bigram counts and
        corresponding unigram counts.
        """

    @abstractmethod
    def _generate_trigram_probs(self):
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

    @abstractmethod
    def _get_bigram_probability(self, bigram):
        """"""

    @abstractmethod
    def _get_trigram_probability(self, trigram):
        """"""

    def _linear_interpolation(self, trigram):
        uni_prob = 0.1 * self.uni_probabilities[trigram[-1]]
        bi_prob = 0.3 * self._get_bigram_probability(trigram[-2:])
        tri_prob = 0.6 * self._get_trigram_probability(trigram)
        return uni_prob + bi_prob + tri_prob

    def text_generator(self, words):
        """
        Generates text based on a given phrase using a language model.

        Args:
            phrase (str): The input phrase to generate text from.
        """
        if not isinstance(words, list):
            words = self._remove_punctuation(words)
            words = words.lower().split()
            words.insert(0, "<s>")

        if len(words) > 1:
            context = tuple(words[-2:])
            loop_prevention_counter = 0

            while context[-1] not in ["</s>", ""] and loop_prevention_counter < 100:
                token_probabilities = {}

                for key in self.tri_probabilities:
                    if key[0:2] == context:
                        token_probabilities[key[-1]] = self._linear_interpolation(key)

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

    def uni_sentence_probability(self, words):
        if not isinstance(words, list):
            words = self._remove_punctuation(words.lower())
            words = words.split()

        sentence_probability = 1
        for unigram in words:
            prob = self.uni_probabilities[unigram]
            sentence_probability *= prob

        return sentence_probability

    def bi_sentence_probability(self, words):
        if not isinstance(words, list):
            words = self._remove_punctuation(words.lower())
            words = ["<s>"] + words.split() + ["</s>"]

        sentence_probability = 1
        for index in range(len(words) - 2):
            bigram = tuple(words[index : index+2])
            prob = self._get_bigram_probability(bigram)
            sentence_probability *= prob

        return sentence_probability

    def tri_sentence_probability(self, words):
        if not isinstance(words, list):
            words = self._remove_punctuation(words.lower())
            words = ["<s>", "<s>"] + words.split() + ["</s>"]

        sentence_probability = 1
        for index in range(len(words) - 3):
            trigram = tuple(words[index : index+3])
            prob = self._get_trigram_probability(trigram)
            sentence_probability *= prob

        return sentence_probability

    def sentence_probability(self, words):
        """
        Calculate the probability of a given sentence according to the language model.

        Parameters:
            sentence (str): The input sentence for which the probability needs to be calculated.

        Returns:
            float: The probability of the given sentence according to the language model.
        """
        if not isinstance(words, list):
            words = self._remove_punctuation(words.lower())
            words = ["<s>", "<s>"] + words.split() + ["</s>"]

        sentence_probability = 1
        for index in range(len(words) - 3):
            trigram = tuple(words[index : index+3])
            prob = self._linear_interpolation(trigram)
            sentence_probability *= prob

        return sentence_probability
