import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import os
from frequency_counts import traverse_tree
from dataset_splitting import split_and_append_elements, write_xml_from_elements

# Creating n_gram counts for the entire corpus
directories = ['aca', 'dem', 'fic', 'news']
BASE_PATH = '../data/corpus/Texts/'

for number_of_words in range(1, 4):
    n_gram_counts = defaultdict(int)

    for directory in directories:
        dir_path = os.path.join(BASE_PATH, directory)
        for file in os.listdir(dir_path):
            if file.endswith('.xml'):
                file_path = os.path.join(dir_path, file)
                tree = ET.parse(file_path)
                root = tree.getroot()
                for child in root:
                    if child.tag != 'teiHeader':
                        traverse_tree(child, number_of_words, n_gram_counts)

    with open(f'n_grams/corpus/{number_of_words}_gram_counts.json', 'w', encoding='utf-8') as fp:
        json.dump(n_gram_counts, fp, indent=4)

# Splitting the corpus into train, validation, and test sets
train = []
validation = []
test = []



for directory in directories:
    dir_path = os.path.join(BASE_PATH, directory)
    for file in os.listdir(dir_path):
        if file.endswith('.xml'):
            file_path = os.path.join(dir_path, file)
            tree = ET.parse(file_path)
            root = tree.getroot()
            sentences = list(root.findall('.//s'))
            split_and_append_elements(sentences, train, validation, test)

write_xml_from_elements(train, '../data/training_set.xml')
write_xml_from_elements(validation, '../data/validation_set.xml')
write_xml_from_elements(test, '../data/test_set.xml')
