import xml.etree.ElementTree as ET
from collections import defaultdict
import json
from frequency_counts import handle_sentence

def generate_counts():
    for number_of_words in range(1, 4):
        n_gram_counts = defaultdict(int)
        tree = ET.parse('../data/training_set.xml')
        root = tree.getroot()
        for child in root:
            handle_sentence(child, number_of_words, n_gram_counts)

        with open(f'n_grams/vanilla/{number_of_words}_gram_counts.json',
                'w', encoding='utf-8') as fp:
            json.dump(n_gram_counts, fp, indent=4)

def uni_gram_prob():
    with open("n_grams/vanilla/1_gram_counts.json", 'r', encoding='utf-8') as fp:
        counts = json.load(fp)

    total_tokens = float(sum(counts.values()))
    probabilities = defaultdict(int)
    for key in counts:
        probabilities[key] = counts[key] / total_tokens

    with open('p_distributions/uni_gram_prob.json', 'w', encoding='utf-8') as fp:
        json.dump(probabilities, fp, indent=4)

def bi_gram_prob():
    with open("n_grams/vanilla/2_gram_counts.json", 'r', encoding='utf-8') as fp:
        bi_gram_counts = json.load(fp)
    with open("n_grams/vanilla/1_gram_counts.json", 'r', encoding='utf-8') as fp:
        uni_gram_counts = json.load(fp)

    probabilities = defaultdict(int)
    for key in bi_gram_counts:
        words = key.split()
        probabilities[key] = bi_gram_counts[key] / uni_gram_counts[words[-1]]

    with open('p_distributions/bi_gram_prob.json', 'w', encoding='utf-8') as fp:
        json.dump(probabilities, fp, indent=4)

def tri_gram_prob():
    with open("n_grams/vanilla/3_gram_counts.json", 'r', encoding='utf-8') as fp:
        tri_gram_counts = json.load(fp)
    with open("n_grams/vanilla/2_gram_counts.json", 'r', encoding='utf-8') as fp:
        bi_gram_counts = json.load(fp)

    probabilities = defaultdict(int)
    for key in tri_gram_counts:
        words = key.split()
        bi_gram_key = words[0] + " " + words[1]
        print(bi_gram_key)
        probabilities[key] = tri_gram_counts[key] / bi_gram_counts[bi_gram_key]

    with open('p_distributions/tri_gram_prob.json', 'w', encoding='utf-8') as fp:
        json.dump(probabilities, fp, indent=4)

uni_gram_prob()
bi_gram_prob()
tri_gram_prob()
