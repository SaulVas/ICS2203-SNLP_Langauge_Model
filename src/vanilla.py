import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import os
from frequency_counts import handle_sentence
from language_model import LanguageModel

class VanillaLM(LanguageModel):
    """
    A language model that generates n-gram counts and calculates probabilities.

    This class represents a language model that generates n-gram counts based on sentences
    in a training set XML file. It calculates the probabilities of unigrams, bigrams, and
    trigrams based on the generated counts.

    Attributes:
        uni_probabilities (defaultdict): A dictionary to store unigram probabilities.
        bi_probabilities (defaultdict): A dictionary to store bigram probabilities.
        tri_probabilities (defaultdict): A dictionary to store trigram probabilities.

    Methods:
        __init__(): Initializes the language model and loads existing 
        n-gram counts or generates new ones.
        __str__(): Returns a string representation of the language model.
        generate_counts(): Generates n-gram counts and saves them to JSON files.
        uni_gram_prob(): Calculates unigram probabilities.
        bi_gram_prob(): Calculates bigram probabilities.
        tri_gram_prob(): Calculates trigram probabilities.
    """

    def _get_counts(self):
        if not (os.path.exists('n_grams/vanilla/1_gram_counts.json')
                and os.path.exists('n_grams/vanilla/2_gram_counts.json')
                and os.path.exists('n_grams/vanilla/3_gram_counts.json')):
            self._generate_counts()
        else:
            with open("n_grams/vanilla/1_gram_counts.json", 'r', encoding='utf-8') as fp:
                self.uni_count = json.load(fp)
            with open("n_grams/vanilla/2_gram_counts.json", 'r', encoding='utf-8') as fp:
                self.bi_count = json.load(fp)
            with open("n_grams/vanilla/3_gram_counts.json", 'r', encoding='utf-8') as fp:
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

            with open(f'n_grams/vanilla/{number_of_words}_gram_counts.json',
                    'w', encoding='utf-8') as fp:
                json.dump(n_gram_counts, fp, indent=4)

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
            self.bi_probabilities[words] = self.bi_count[key] / self.uni_count[words[-1]]

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

vanilla = VanillaLM()
vanilla.text_generator("to live")