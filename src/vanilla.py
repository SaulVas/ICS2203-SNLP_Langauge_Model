import xml.etree.ElementTree as ET
from collections import defaultdict
import json
from frequency_counts import handle_sentence

for number_of_words in range(1, 4):
    n_gram_counts = defaultdict(int)
    tree = ET.parse('../data/training_set.xml')
    root = tree.getroot()
    for child in root:
        handle_sentence(child, number_of_words, n_gram_counts)

    with open(f'n_grams/vanilla/{number_of_words}_gram_counts.json', 'w', encoding='utf-8') as fp:
        json.dump(n_gram_counts, fp, indent=4)
