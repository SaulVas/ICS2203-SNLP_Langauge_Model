import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import os
from frequency_counts import traverse_tree, retrieve_text
from dataset_management import split_and_append_elements, write_xml_from_elements
from vanilla import VanillaLM
from laplace import LaplaceLM
from unk import UnkLM

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

TRAIN_FILE_PATH = '../data/training_set.xml'
TEST_FILE_PATH = '../data/test_set.xml'
if not (os.path.exists(TRAIN_FILE_PATH)
        and os.path.exists(TEST_FILE_PATH)):
# Splitting the corpus into train, validation, and test sets if not already created
    train = []
    test = []

    for directory in directories:
        dir_path = os.path.join(BASE_PATH, directory)
        for file in os.listdir(dir_path):
            if file.endswith('.xml'):
                file_path = os.path.join(dir_path, file)
                tree = ET.parse(file_path)
                root = tree.getroot()
                sentences = list(root.findall('.//s'))
                split_and_append_elements(sentences, train, test)

    if os.path.exists(TRAIN_FILE_PATH):
        os.remove(TRAIN_FILE_PATH)
    if os.path.exists(TEST_FILE_PATH):
        os.remove(TEST_FILE_PATH)
    write_xml_from_elements(train, TRAIN_FILE_PATH)
    write_xml_from_elements(test, TEST_FILE_PATH)

Vanilla = VanillaLM()
laplace = LaplaceLM()
unk = UnkLM()

test_sentences = []
test_tree = ET.parse("../data/test_set.xml")
root = test_tree.getroot()
for child in root:
    test_sentences.append(retrieve_text(child))

# Vanilla
total_uni_log_prob = 0
total_bi_log_prob = 0
total_tri_log_prob = 0
for sentence in test_sentences:

    
# Laplace
    
# Unk
