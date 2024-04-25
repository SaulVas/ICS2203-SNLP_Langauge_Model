import xml.etree.ElementTree as ET
import json
import os
from dataset_functions import (retrieve_text, model_perplexity,
                               generate_corpus_counts, splitting_datasets)
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

def text_generation(models):
    print("starting up text generation")
    while True:
        model_choice = input("Please choose a model:\n"
                             + "Vanilla: 1\n"
                             + "Laplace: 2\n"
                             + "UNK: 3\n"
                             + "or press q to quit\n").strip()

        while model_choice not in ['1', '2', '3', 'q']:
            print("Invalid input, try again")
            model_choice = input()

        if model_choice == 'q':
            break

        ngram_choice = input("Please choose a model:\n"
                             + "unigram: 1\n"
                             + "bigram: 2\n"
                             + "trigram: 3\n"
                             + "linear interpolation: 4\n").strip()

        while model_choice not in ['1', '2', '3', '4']:
            print("Invalid input, try again")
            ngram_choice = input()

        sentence = input("Input a phrase to be finished by your selected model\n")
        if model_choice == '1':
            models[0].text_generator(sentence, ngram_choice)
        elif model_choice == '2':
            models[1].text_generator(sentence, ngram_choice)
        elif model_choice == '3':
            models[2].text_generator(sentence, ngram_choice)

def sentence_probability_calculator(models):
    print("starting up sentence probability calculator")
    while True:
        model_choice = input("Please choose a model:\n"
                             + "Vanilla: 1\n"
                              + "Laplace: 2\n"
                             + "UNK: 3\n"
                             + "or press q to quit\n").strip()

        while model_choice not in ['1', '2', '3', 'q']:
            print("Invalid input, try again")
            model_choice = input()

        if model_choice == 'q':
            break

        sentence = input("Input a sentence\n")
        if model_choice == '1':
            print("The probability of your sentence is: "
                  + models[0].sentence_probability(sentence))
        elif model_choice == '2':
            print("The probability of your sentence is: "
                  + models[1].sentence_probability(sentence))
        elif model_choice == '3':
            print("The probability of your sentence is: "
                  + models[2].sentence_probability(sentence))

if __name__ == "__main__":
    if not (os.path.exists("n_grams/corpus/1_gram_counts.json")
            and os.path.exists("n_grams/corpus/2_gram_counts.json")
            and os.path.exists("n_grams/corpus/3_gram_counts.json")):
        print("Generating corpus counts...")
        generate_corpus_counts()

    if not (os.path.exists('../data/training_set.xml')
            and os.path.exists('../data/test_set.xml')):
        print("Splitting the data sets...")
        splitting_datasets()

    print("Training the models...")
    vanilla = VanillaLM()
    laplace = LaplaceLM()
    unk = UnkLM()
    lms = [vanilla, laplace, unk]

    while True:
        function_choice = input("Please choose a function:\n"
                             + "Text Generation: 1\n"
                             + "Sentence Probability Calculator: 2\n"
                             + "or press q to quit\n").strip()

        while function_choice not in ['1', '2', 'q']:
            print("Invalid input, try again")
            function_choice = input()

        if function_choice == '1':
            text_generation(lms)
        elif function_choice == '2':
            sentence_probability_calculator(lms)
        else:
            break
