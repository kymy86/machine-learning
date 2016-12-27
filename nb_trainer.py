#!/usr/bin/python3

from __future__ import division
import re
import pickle
import os
from pathlib import Path
from math import log10
from logger import Logger


class Trainer(Logger):

    _DATASET = 'dataset/SMSSpamCollection'
    STOREDATA = 'dataset/MemoryData'
    _TRESHOLD = 0.5
    _dataset = list()
    words_counter = list()
    dictionary = list()
    m_spam = 0
    m_ham = 0
    m_tot = 0
    n_words = 0
    offset = 0
    w_spam = 0
    w_ham = 0

    def __init__(self):
        super(Trainer, self).__init__()
        if not self._is_memory():
            self._load_data()
            self._compute_dict()
            self._init_params()

    def _load_data(self):
        """
         Load the sample data to train the
         algorithm
        """
        with open(self._DATASET, 'r') as dataset_filename:
            lines = dataset_filename.readlines()
            for line in lines:
                self._dataset.append(line.strip("\n").split("\t"))

    def _compute_dict(self):
        for document in self._dataset:
            words = self.get_list_words(document[1])
            self.words_counter.append({i:words.count(i) for i in words})
            for word in words:
                if word not in self.dictionary:
                    self.dictionary.append({'p_spam':1, 'p_ham':1, 'word':word})

    def _init_params(self):
        for item in self._dataset:
            if item[0] == 'spam':
                self.m_spam += 1
            else:
                self.m_ham += 1
        self.m_tot = len(self._dataset)
        self.n_words = len(self.dictionary)
        self.w_spam = self.n_words
        self.w_ham = self.n_words
        self.offset = log10(self._TRESHOLD)+log10(self.m_ham)-log10(self.m_spam)

    @staticmethod
    def get_list_words(document):
        """
        From the given document, create a list of
        allow words. Remove the single char words
        """
        regex = '(the|to|you|he|she|only|if|it|[.,#!?]|)'
        words = re.sub(regex, '', document, flags=re.IGNORECASE).lower().split()
        for word in words:
            if re.match('[a-z0-9]{1}', word, flags=re.IGNORECASE):
                words.remove(word)
        return words

    def train_agent(self):
        """
        From the dictionary loaded,
        start to train the agent
        """
        if not self._is_memory():
            for idx, line in enumerate(self._dataset):
                if line[0] == 'spam':
                    self._compute_prob('p_spam', idx)
                else:
                    self._compute_prob('p_ham', idx)
            self._normalization()
            self._store_samples_in_memory()

    def _normalization(self):
        for word in self.dictionary:
            word['p_ham'] = word['p_ham']/self.w_ham
            word['p_spam'] = word['p_spam']/self.w_spam


    def _compute_prob(self, key, idx):
        for word in self.dictionary:
            if word['word'] in self.words_counter[idx]:
                counter = self.words_counter[idx][word['word']]
            else:
                counter = 0
            word[key] = word[key] + counter
            if key == 'p_spam':
                self.w_spam = self.w_spam + counter
            else:
                self.w_ham = self.w_ham + counter

    def _store_samples_in_memory(self):
        pickle.dump(self.dictionary, open(self.STOREDATA, 'wb'))

    def _is_memory(self):
        memory_name = Path(self.STOREDATA)
        return memory_name.is_file()

    def reset_memory(self):
        """
        Reset memory and re-train the agent
        """
        if self._is_memory():
            os.remove(self.STOREDATA)

