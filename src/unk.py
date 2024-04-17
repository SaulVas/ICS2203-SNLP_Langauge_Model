"""
Implementation of the unk language model.
"""
import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import os
from vanilla import VanillaLM
from frequency_counts import handle_sentence, handle_sentence_unk


class UnkLM(VanillaLM):
    def __init__(self):
        self.unknown_tokens = set()
        super().__init__()

    def _defualt_uni_value(self):
        return float(1 / sum(self.uni_count.values()) + len(self.uni_count))

    def _get_counts(self):
        """
        Loads the n-gram counts from JSON files if they exist, otherwise generates the counts.

        If the JSON files for 1-gram, 2-gram, and 3-gram counts exist in the 'n_grams/unk'
        directory, this method loads the counts from the files and assigns them to the 
        corresponding instance variables. If the files do not exist, it calls the 
        '_generate_counts' method to generate the counts.

        Args:
            None

        Returns:
            None
        """
        if not (os.path.exists('n_grams/unk/1_gram_counts.json')
                and os.path.exists('n_grams/unk/2_gram_counts.json')
                and os.path.exists('n_grams/unk/3_gram_counts.json')):
            self._generate_counts()

        with open("n_grams/unk/1_gram_counts.json", 'r', encoding='utf-8') as fp:
            self.uni_count = json.load(fp)
        with open("n_grams/unk/2_gram_counts.json", 'r', encoding='utf-8') as fp:
            self.bi_count = json.load(fp)
        with open("n_grams/unk/3_gram_counts.json", 'r', encoding='utf-8') as fp:
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
        n_gram_counts = defaultdict(int)
        tree = ET.parse('../data/training_set.xml')
        root = tree.getroot()
        for child in root:
            handle_sentence(child, 1, n_gram_counts)

        self.unknown_tokens = {key for key, count in n_gram_counts.items() if count <= 2}

        # Generate real counts:
        for number_of_words in range(1, 4):
            n_gram_counts = defaultdict(int)
            for child in root:
                handle_sentence_unk(child, number_of_words, n_gram_counts, self.unknown_tokens)

            with open(f'n_grams/unk/{number_of_words}_gram_counts.json',
                    'w', encoding='utf-8') as fp:
                json.dump(n_gram_counts, fp, indent=4)

        print(self.uni_count["<UNK>"])

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

    def text_generator(self, words):
        words = self._remove_punctuation(words)
        words = words.lower().split()
        words.insert(0, "<s>")
        for word, index in enumerate(words):
            if word in self.unknown_tokens:
                words[index] = "<UNK>"
        return super().text_generator(words)

    def sentence_probability(self, words):
        words = self._remove_punctuation(words.lower())
        words = ["<s>", "<s>"] + words.split() + ["</s>"]
        for word, index in enumerate(words):
            if word in self.unknown_tokens:
                words[index] = "<UNK>"
        return super().sentence_probability(words)

