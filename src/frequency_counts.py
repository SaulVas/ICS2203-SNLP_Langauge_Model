"""
This module contains functions for processing sentences and computing n-gram frequencies.

Author: [Saul Vassallo]
Date: [8th april 2024]
"""

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
                print("set UNK")
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
