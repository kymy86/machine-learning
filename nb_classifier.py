#!/usr/bin/python3

from math import log10
import pickle
import sys
from pathlib import Path
from logger import Logger
from nb_trainer import Trainer

class Classifier(Logger):

    treshold = 0
    dictionary = list()
    test_data = list()

    def __init__(self, offset, training_dataset, test_dataset):
        super(Classifier, self).__init__()
        # initialize score treshold
        self.treshold = -offset
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self._load_memory()

    def _load_memory(self):
        memory_name = Path(self.training_dataset)
        if not memory_name.is_file():
            super(Classifier, self).debug("Memory data not exist. Train the agent before")
            sys.exit(1)
        else:
            with open(self.training_dataset, 'rb') as dataset:
                self.dictionary = pickle.load(dataset)
            with open(self.test_dataset, 'rb') as test_data:
                self.test_data = pickle.load(test_data)

    def classify(self, document):
        """
        Run Naive Bayesian classifier
        """

        words = Trainer.get_list_words(document)
        word_dict = {i:words.count(i) for i in words}

        for word in self.dictionary:
            if word['word'] in word_dict:
                #product could lead to numerical overflow, use sum of logarithm instead
                elem = word_dict[word['word']]*(log10(word['p_spam'])-log10(word['p_ham']))
                self.treshold = self.treshold + elem

    def is_spam(self):
        """
        If treshold is positive than is a spam
        else is ham
        """
        return self.treshold > 0


