"""
This module contains functions for processing sentences and computing n-gram frequencies.

Author: [Saul Vassallo]
Date: [8th april 2024]
"""

import random
import xml.etree.ElementTree as ET
import numpy as np

random.seed(42)

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
    total_uni_log_prob = 0
    total_bi_log_prob = 0
    total_tri_log_prob = 0
    total_lin_log_prob = 0

    for sentence in sentences:
        uni_prob = model.uni_sentence_probability(sentence)
        total_uni_log_prob += np.log(uni_prob)

        bi_prob = model.bi_sentence_probability(sentence)
        total_bi_log_prob += np.log(bi_prob)

        tri_prob = model.tri_sentence_probability(sentence)
        total_tri_log_prob += np.log(tri_prob)

        lin_prob = model.sentence_probability(sentence)
        total_lin_log_prob += np.log(lin_prob)

    average_uni_log_prob = total_uni_log_prob / len(sentences)
    average_bi_log_prob = total_bi_log_prob / len(sentences)
    average_tri_log_prob = total_tri_log_prob / len(sentences)
    average_lin_log_prob = total_lin_log_prob / len(sentences)

    unigram_perplexity = np.power(2, -average_uni_log_prob)
    bigram_perplexity = np.power(2, -average_bi_log_prob)
    trigram_perplexity = np.power(2, -average_tri_log_prob)
    linear_interpolation_perplexity = np.power(2, -average_lin_log_prob)

    return (unigram_perplexity, bigram_perplexity,
            trigram_perplexity, linear_interpolation_perplexity)
