"""
This module provides a function for splitting a list of elements into 
train, validation, and test sets.
"""
import random
import xml.etree.ElementTree as ET
import os

random.seed(42)

def split_and_append_elements(s_elements, all_train_elements, all_validation_elements, all_test_elements):
    """
    Splits the given list of elements into train, validation, and test sets,
    and appends them to the respective global lists.

    Args:
        s_elements (list): The list of elements to be split.

    Returns:
        None
    """
    total_elements = len(s_elements)
    train_size = int(total_elements * 0.7)
    validation_size = int(total_elements * 0.2)

    random.shuffle(s_elements)
    train_elements = s_elements[:train_size]
    validation_elements = s_elements[train_size:train_size+validation_size]
    test_elements = s_elements[train_size+validation_size:]

    all_train_elements.extend(train_elements)
    all_validation_elements.extend(validation_elements)
    all_test_elements.extend(test_elements)

def write_xml_from_elements(elements, path):
    """
    Write XML file from a list of elements.

    Args:
        elements (list): List of XML elements to be included in the XML file.
        path (str): Path to the output XML file.

    Returns:
        None
    """
    new_root = ET.Element('root')
    new_root.extend(elements)
    new_tree = ET.ElementTree(new_root)
    new_tree.write(path)
