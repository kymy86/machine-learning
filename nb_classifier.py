#!/usr/bin/python3

from math import log10
import pickle
from pathlib import Path
from logger import Logger
from nb_trainer import Trainer

class Classifier(Logger):

    treshold = 0
    dictionary = list()

    def __init__(self, offset):
        super(Classifier, self).__init__()
        self.treshold = -offset
        self._load_memory()

    def _load_memory(self):
        memory_name = Path(Trainer.STOREDATA)
        if not memory_name.is_file():
            super(Classifier, self).debug("Memory data not exist. Train the agent before")
        else:
            with open(Trainer.STOREDATA, 'rb') as dataset:
                self.dictionary = pickle.load(dataset)

    def classify(self, document):
        """
        Run Naive Bayesian classifier
        """
        words = Trainer.get_list_words(document)
        word_dict = {i:words.count(i) for i in words}

        for word in self.dictionary:
            if word['word'] in word_dict:
                elem = word_dict[word['word']]*(log10(word['p_spam'])-log10(word['p_ham']))
            else:
                elem = 0
            self.treshold = self.treshold + elem

    def is_spam(self):
        """
        If treshold is positive than is a spam
        else is ham
        """
        return self.treshold > 0


