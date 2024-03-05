import xml.etree.ElementTree as ET
from collections import defaultdict 


def frequency_counts():
    tree = ET.parse('../../corpus/Texts/aca/A6U.xml')
    root = tree.getroot()
    wText = root.find('.//wtext')
    traverse_tree(wText)

def traverse_tree(node):
    for child in node:
        if child.tag == 's':
            handle_sentence(child)
        else:
            traverse_tree(child)

def handle_sentence(sentence_node):
    text = retrieve_text(sentence_node)
    if text != "":
        

def retrieve_text(node):
    text = ""
    for child in node:
        if child.tag == 'w':
            text += child.text.lower()
        else:
            if len(node) > 0:
                for grandchild in child:
                    text += retrieve_text(grandchild)
                
    return text

frequency_counts()
