import xml.etree.ElementTree as ET
import json
from dataset_functions import retrieve_text, model_perplexity
# from dataset_functions import generate_corpus_counts, splitting_datasets
from vanilla import VanillaLM
from laplace import LaplaceLM
from unk import UnkLM

def calculate_perplexities(models):
    test_sentences = []
    test_tree = ET.parse("../documentation/test_1.xml")
    root = test_tree.getroot()
    for child in root:
        test_sentences.append(retrieve_text(child))

    perplexities = {}

    for model in models:
        perplexities[model.__class__.__name__] = model_perplexity(model, test_sentences)

    with open('../documentation/perplexity.json', 'w', encoding='utf-8') as fp:
        json.dump(perplexities, fp, indent=4)

if __name__ == "__main__":
    # generate_corpus_counts()
    # splitting_datasets()

    vanilla = VanillaLM()
    laplace = LaplaceLM()
    unk = UnkLM()

    lms = [vanilla, laplace, unk]
    calculate_perplexities(lms)

    