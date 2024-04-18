import xml.etree.ElementTree as ET
from vanilla import VanillaLM
from dataset_functions import retrieve_text, model_perplexity

vanilla = VanillaLM()

test_sentences = []
test_tree = ET.parse("../documentation/test_1.xml")
root = test_tree.getroot()
for child in root:
    test_sentences.append(retrieve_text(child))

perplexities = {}

perplexities["vanilla"] = model_perplexity(vanilla, test_sentences)