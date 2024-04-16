"""
Implementation of the unk language model.
"""
import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import os
from frequency_counts import handle_sentence
from language_model_abc import LanguageModel

class Unk(LanguageModel):
    def _defualt_uni_value(self):
        return float

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

        # Set threshold for unknown and go through training
        # corpus changing words below threshold to <UNK>
        # ...

        # Generate real counts:
        for number_of_words in range(1, 4):
            n_gram_counts = defaultdict(int)
            tree = ET.parse('../data/training_set.xml')
            root = tree.getroot()
            for child in root:
                handle_sentence(child, number_of_words, n_gram_counts)

            with open(f'n_grams/unk/{number_of_words}_gram_counts.json',
                    'w', encoding='utf-8') as fp:
                json.dump(n_gram_counts, fp, indent=4)
