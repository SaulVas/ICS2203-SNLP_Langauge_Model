""" Calculate and store n-gram frequencies from XML text data.

This module is designed to process text data contained within XML files,
specifically targeting text wrapped within specified XML tags. It computes
frequency counts for unigrams, bigrams, and trigrams found in the text. The
frequencies are calculated for each sentence, considering the start (`<s>`)
and end (`</s>`) of sentences to better model sentence boundaries.

The results are saved in JSON format, with separate files for unigrams,
bigrams, and trigrams. These files can be used for further analysis, such as
language modeling or statistical analysis of the text.

Functions:
- frequency_counts: Main function to parse XML, compute n-gram frequencies,
  and save the results to JSON files.
- traverse_tree: Recursively traverses the XML tree to find sentences and
  process their text for n-gram frequency calculation.
- handle_sentence: Processes each sentence to compute and update n-gram
  frequencies.
- retrieve_text: Extracts and concatenates text from XML nodes, adding start
  and end markers to each sentence.

The module uses `xml.etree.ElementTree` for XML parsing, `collections.defaultdict`
for counting n-gram frequencies, and `json` for saving the results.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
import json

def frequency_counts():
    """ Computes and saves n-gram frequencies from an XML text file.

    This function parses an XML file, extracts text from a specified node,
    and calculates the frequencies of unigrams, bigrams, and trigrams within
    the text. The computed frequencies are saved in separate JSON files,
    one for each n-gram type, with filenames indicating the n-gram size.

    Assumes the text is contained within a 'wtext' node in the XML structure.
    The output JSON files are saved to a relative directory '../n_grams/'.

    No parameters are taken, and the XML file path is hard-coded. The function
    is designed to work with a specific XML structure and file location.
    """
    tree = ET.parse('../../corpus/Texts/aca/A6U.xml')
    root = tree.getroot()
    text_node = root.find('.//wtext')

    # Create frequency counts for unigrams, bigrams, and trigrams and save to a json file
    for number_of_words in range(1, 4):
        n_gram_counts = defaultdict(int)
        traverse_tree(text_node, number_of_words, n_gram_counts)

        with open(f'../n_grams/{number_of_words}_gram_counts.json', 'w', encoding='utf-8') as fp:
            json.dump(n_gram_counts, fp)


def traverse_tree(node, number_of_words, counts):
    """ Recursively traverses the XML tree to find sentences and process their 
    text for n-gram frequency calculation.

    Parameters:
    node (xml.etree.ElementTree.Element): The current node in the XML tree.
    number_of_words (int): The number of words in the n-grams to be counted.
    counts (collections.defaultdict): A dictionary to store the n-gram counts.
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
    """
    text = retrieve_text(sentence_node) + " </s>"
    if text != "<s> ":
        words = text.split()
        for index in range(len(words) - number_of_words + 1):
            if number_of_words == 1:
                n_gram = words[index]
            else:
                n_gram = " ".join(words[index:index + number_of_words]) 
            counts[n_gram] += 1


def retrieve_text(node):
    """ Extracts and concatenates text from XML nodes, adding start and end markers to each sentence.

    Parameters:
    node (xml.etree.ElementTree.Element): The current node in the XML tree.

    Returns:
    str: The concatenated text from the node and its descendants.
    """
    text = "<s> "
    for child in node:
        if child.tag == 'w':
            text += child.text.lower()
        else:
            if len(node) > 0:
                for grandchild in child:
                    text += retrieve_text(grandchild) 
    return text


frequency_counts()
