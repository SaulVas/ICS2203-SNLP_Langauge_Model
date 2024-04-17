"""
This module contains functions for splitting arrays into a training, 
validation and test set and storing them in xml files.

Author: [Saul Vassallo]
Date: [8th april 2024]
"""

import random
import xml.etree.ElementTree as ET

random.seed(42)

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
