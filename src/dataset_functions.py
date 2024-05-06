"""
This module contains functions for processing sentences and computing n-gram frequencies.

Author: [Saul Vassallo]
Date: [8th april 2024]
"""

import random
import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np

random.seed(42)

directories = ['aca', 'dem', 'fic', 'news']
BASE_PATH = '../data/corpus/Texts/'

def generate_corpus_counts():
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

        with open(f'n_grams/corpus/{number_of_words}_gram_counts.json',
                  'w', encoding='utf-8') as fp:
            json.dump(n_gram_counts, fp, indent=4)

def traverse_tree(node, number_of_words, counts):
    """ Recursively traverses the XML tree to find sentences and process their 
    text for n-gram frequency calculation.

    Parameters:
    node (xml.etree.ElementTree.Element): The current node in the XML tree.
    number_of_words (int): The number of words in the n-grams to be counted.
    counts (collections.defaultdict): A dictionary to store the n-gram counts.

    Returns:
        None
    """
    for child in node:
        if child.tag == 's':
            handle_sentence(child, number_of_words, counts)
        else:
            traverse_tree(child, number_of_words, counts)

def handle_sentence(sentence_node, number_of_words, counts):
    """ Processes each sentence to compute and update n-gram frequencies.

    Parameters:
    sentence_node (xml.etree.ElementTree.Element): The current sentence node in the XML tree.
    number_of_words (int): The number of words in the n-grams to be counted.
    counts (collections.defaultdict): A dictionary to store the n-gram counts.
    
    Returns:
        None
    """
    text = retrieve_text(sentence_node)
    if text.strip() != "":
        text = ("<s> " * number_of_words) + text + (" </s>")
        words = text.split()
        for index in range(len(words) - number_of_words + 1):
            if number_of_words == 1:
                n_gram = words[index]
            else:
                n_gram = " ".join(words[index:index + number_of_words])
            counts[n_gram] += 1

def handle_sentence_unk(sentence_node, number_of_words, counts, unknown_tokens):
    """ Processes each sentence to compute and update n-gram frequencies.

    Parameters:
    sentence_node (xml.etree.ElementTree.Element): The current sentence node in the XML tree.
    number_of_words (int): The number of words in the n-grams to be counted.
    counts (collections.defaultdict): A dictionary to store the n-gram counts.
    
    Returns:
        None
    """
    text = retrieve_text(sentence_node)
    if text.strip() != "":
        text = ("<s> " * number_of_words) + text + (" </s>")
        words = text.split()
        for index in range(len(words) - number_of_words + 1):
            if words[index] in unknown_tokens:
                words[index] = "<UNK>"
            if number_of_words == 1:
                n_gram = words[index]
            else:
                n_gram = " ".join(words[index:index + number_of_words])
            counts[n_gram] += 1

def retrieve_text(node):
    """ Extracts and concatenates text from XML nodes, adding start and end 
    markers to each sentence.

    Parameters:
    node (xml.etree.ElementTree.Element): The current node in the XML tree.

    Returns:
    str: The concatenated text from the node and its descendants.
    """
    text = ""
    for child in node:
        if child.tag == 'w':
            if child.text:
                text += child.text.lower()
        elif child.tag == 'c':
            text += " "
        else:
            if len(node) > 0:
                for grandchild in child:
                    text += retrieve_text(grandchild)
    return text

def splitting_datasets():
    train_file_path = '../data/training_set.xml'
    test_file_path = '../data/test_set.xml'
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

    if os.path.exists(train_file_path):
        os.remove(train_file_path)
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
    write_xml_from_elements(train, train_file_path)
    write_xml_from_elements(test, test_file_path)

def split_and_append_elements(s_elements, training_set, test_set):
    """
    Splits the given list of elements into train, test, and test sets,
    and appends them to the respective global lists.

    Args:
        s_elements (list): The list of elements to be split.
        training_set (list): The list to append the training set elements to.
        test_set (list): The list to append the test set elements to.
        test_set (list): The list to append the test set elements to.

    Returns:
        None
    """
    total_elements = len(s_elements)
    train_size = int(total_elements * 0.8)

    random.shuffle(s_elements)
    train_elements = s_elements[:train_size]
    test_elements = s_elements[train_size:]

    training_set.extend(train_elements)
    test_set.extend(test_elements)

def write_xml_from_elements(elements, path):
    """
    Write XML file from a list of elements.

    Args:
        elements (list): List of XML elements to be included in the XML file.
        path (str): Path to the output XML file.

    Returns:
        None
    """
    root = ET.Element('root')
    root.extend(elements)
    tree = ET.ElementTree(root)
    tree.write(path)


def model_perplexity(model, sentences):
    total_uni_pp = 0
    total_bi_pp = 0
    total_tri_pp = 0
    total_lin_pp = 0

    for sentence in sentences:
        if len(sentence.split()) == 0:
            continue
        uni_prob = model.uni_sentence_probability(sentence)
        bi_prob = model.bi_sentence_probability(sentence)
        tri_prob = model.tri_sentence_probability(sentence)
        lin_prob = model.sentence_probability(sentence)

        if uni_prob == 0 or bi_prob == 0 or tri_prob == 0 or lin_prob == 0:
            uni_prob = model.uni_sentence_probability(sentence)
            bi_prob = model.bi_sentence_probability(sentence)
            tri_prob = model.tri_sentence_probability(sentence)
            lin_prob = model.sentence_probability(sentence)

        uni_pp = np.power(np.divide(1, uni_prob), (1 / len(sentence.split())))
        total_uni_pp += uni_pp

        bi_pp = np.power(np.divide(1, bi_prob), (1 / len(sentence.split())))
        total_bi_pp += bi_pp

        tri_pp = np.power(np.divide(1, tri_prob), (1 / len(sentence.split())))
        total_tri_pp += tri_pp

        lin_pp = np.power(np.divide(1, lin_prob), (1 / len(sentence.split())))
        total_lin_pp += lin_pp

    average_uni_pp = total_uni_pp / len(sentences)
    average_bi_pp = total_bi_pp / len(sentences)
    average_tri_pp = total_tri_pp / len(sentences)
    average_lin_pp = total_lin_pp / len(sentences)

    return (average_uni_pp, average_bi_pp, average_tri_pp, average_lin_pp)
