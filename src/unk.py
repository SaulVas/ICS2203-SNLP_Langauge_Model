"""
Implementation of the unk language model.
"""
import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import os
import sys
from vanilla import VanillaLM
from dataset_functions import handle_sentence, handle_sentence_unk

class UnkLM(VanillaLM):
    def __init__(self):
        super().__init__()
        self.vocabulary = set(self.uni_count)

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

        unknown_tokens = {key for key, count in n_gram_counts.items() if count <= 2}

        # Generate real counts:
        for number_of_words in range(1, 4):
            n_gram_counts = defaultdict(int)
            for child in root:
                handle_sentence_unk(child, number_of_words, n_gram_counts, unknown_tokens)

            with open(f'n_grams/unk/{number_of_words}_gram_counts.json',
                    'w', encoding='utf-8') as fp:
                json.dump(n_gram_counts, fp, indent=4)

    def _generate_unigram_probs(self):
        total_tokens = float(sum(self.uni_count.values()))
        for key in self.uni_count:
            self.uni_probabilities[key] = ((self.uni_count[key] + 1)
                                           / (total_tokens + len(self.uni_count)))

    def _generate_bigram_probs(self):
        for key in self.bi_count:
            words = tuple(key.split())
            self.bi_probabilities[words] = ((self.bi_count[key] + 1)
                                            / (self.uni_count[words[0]] + len(self.uni_count)))

    def _generate_trigram_probs(self):
        for key in self.tri_count:
            words = tuple(key.split())
            bi_gram_key = words[0] + " " + words[1]
            self.tri_probabilities[words] = ((self.tri_count[key] + 1)
                                             / (self.bi_count[bi_gram_key] + len(self.uni_count)))

    def _get_bigram_probability(self, bigram):
        return self.bi_probabilities.get(bigram,
                                         1 / (self.uni_count.get(bigram[0], 1)
                                              + len(self.uni_count)))

    def _get_trigram_probability(self, trigram):
        return self.tri_probabilities.get(trigram,
                                          1 / (self.bi_count.get(trigram[:2], 1)
                                               + len(self.uni_count)))

    def text_generator(self, sentence, choice):
        sentence = self._remove_punctuation(sentence)
        sentence = sentence.lower().split()
        for index, word in enumerate(sentence):
            if word not in self.vocabulary:
                sentence[index] = "<UNK>"
        return super().text_generator(sentence, choice)

    def uni_sentence_probability(self, words):
        words = self._remove_punctuation(words.lower())
        words = words.split()
        for index, word in enumerate(words):
            if word not in self.vocabulary:
                words[index] = "<UNK>"
        return max(super().uni_sentence_probability(words), sys.float_info.min)

    def bi_sentence_probability(self, words):
        words = self._remove_punctuation(words.lower())
        words = ["<s>"] + words.split() + ["</s>"]
        for index, word in enumerate(words):
            if word not in self.vocabulary:
                words[index] = "<UNK>"
        return max(super().bi_sentence_probability(words), sys.float_info.min)

    def tri_sentence_probability(self, words):
        words = self._remove_punctuation(words.lower())
        words = ["<s>", "<s>"] + words.split() + ["</s>"]
        for index, word in enumerate(words):
            if word not in self.vocabulary:
                words[index] = "<UNK>"
        return max(super().tri_sentence_probability(words), sys.float_info.min)

    def sentence_probability(self, words):
        words = self._remove_punctuation(words.lower())
        words = ["<s>", "<s>"] + words.split() + ["</s>"]
        for index, word in enumerate(words):
            if word not in self.vocabulary:
                words[index] = "<UNK>"
        return max(super().sentence_probability(words), sys.float_info.min)

    def calculate_space_needed(self):
        size = sys.getsizeof(self.uni_count)        
        for key, value in self.uni_count.items():
            size += sys.getsizeof(key)
            size += sys.getsizeof(value)

        size += sys.getsizeof(self.bi_count)
        for key, value in self.bi_count.items():
            size += sys.getsizeof(key)
            size += sys.getsizeof(value)

        size += sys.getsizeof(self.tri_count)
        for key, value in self.tri_count.items():
            size += sys.getsizeof(key)
            size += sys.getsizeof(value)

        size += sys.getsizeof(self.uni_probabilities)
        for key, value in self.uni_probabilities.items():
            size += sys.getsizeof(key)
            size += sys.getsizeof(value)

        size += sys.getsizeof(self.bi_probabilities)
        for key, value in self.bi_probabilities.items():
            size += sys.getsizeof(key)
            size += sys.getsizeof(value)

        size += sys.getsizeof(self.tri_probabilities)
        for key, value in self.tri_probabilities.items():
            size += sys.getsizeof(key)
            size += sys.getsizeof(value)

        size += sys.getsizeof(self.vocabulary)
        for key in self.vocabulary:
            size += sys.getsizeof(key)
            
        print(size)