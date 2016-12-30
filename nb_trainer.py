#!/usr/bin/python3

from __future__ import division
import re
import pickle
import os
from random import randrange
from pathlib import Path
from math import log10
from logger import Logger


class Trainer(Logger):

    _DATASET = 'dataset/SMSSpamCollection'
    STOREDATA = 'dataset/MemoryTrainingData'
    STORETESTDATA = 'dataset/MemoryTestData'
    #rejection treshold
    _TRESHOLD = 100
    # document dataset
    _dataset = list()
    # list of test data
    test_data = list()
    #occurences of each word in each document
    words_counter = list()
    # dictionary with all words in dataset and spam/ham probability
    dictionary = list()
    # tot number of spam
    m_spam = 0
    # tot number of ham
    m_ham = 0
    # tot number of documents
    m_tot = 0
    n_words = 0
    offset = 0
    w_spam = 0
    #tot number of ham
    w_ham = 0

    def __init__(self, dataset_size=0.7):
        super(Trainer, self).__init__()
        if not self._is_memory():
            self._load_data(dataset_size)
            self._compute_dict()
            self._init_params()

    def _load_data(self, split_ratio):
        """
         Load the sample data to train the
         algorithm
        """
        with open(self._DATASET, 'r') as dataset_filename:
            lines = dataset_filename.readlines()
            for line in lines:
                self._dataset.append(line.strip("\n").split("\t"))
        self._compute_train_test_dataset(split_ratio)

    def _compute_train_test_dataset(self, split_ratio):
        dataset_size = int(len(self._dataset)*split_ratio)
        train_set = []
        test_set = list(self._dataset)
        while len(train_set) < dataset_size:
            index = randrange(len(test_set))
            train_set.append(test_set.pop(index))
        self._dataset = train_set
        self.test_data = test_set


    def _compute_dict(self):
        for document in self._dataset:
            words = self.get_list_words(document[1])
            # count the occurences of each word in each document
            self.words_counter.append({i:words.count(i) for i in words})
            for word in words:
                if word not in self.dictionary:
                    #add words in the dictionary, by initializing them with score of 1
                    self.dictionary.append({'p_spam':1, 'p_ham':1, 'word':word})

    def _init_params(self):
        #count the number of spam/ham documents
        for item in self._dataset:
            if item[0] == 'spam':
                self.m_spam += 1
            else:
                self.m_ham += 1
        self.m_tot = len(self._dataset)
        self.n_words = len(self.dictionary)
        self.w_spam = self.n_words
        self.w_ham = self.n_words
        # offset the rejection treshold
        self.offset = log10(self._TRESHOLD)+log10(self.m_ham)-log10(self.m_spam)

    @staticmethod
    def get_list_words(document):
        """
        From the given document, create a list of
        allow words. Remove the single char words
        an the most frequent words
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
                #assign a score to each word
                if line[0] == 'spam':
                    self._count_occurences('p_spam', idx)
                else:
                    self._count_occurences('p_ham', idx)
            self._normalization()
            self._store_samples_in_memory()

    def _normalization(self):
        #normalize count to get the probabilities
        for word in self.dictionary:
            word['p_ham'] = word['p_ham']/self.w_ham
            word['p_spam'] = word['p_spam']/self.w_spam


    def _count_occurences(self, key, idx):
        for word in self.dictionary:
            if word['word'] in self.words_counter[idx]:
                # counter is the number of times a given word appear in the given document
                counter = self.words_counter[idx][word['word']]
                word[key] = word[key] + counter
                if key == 'p_spam':
                    self.w_spam = self.w_spam + counter
                else:
                    self.w_ham = self.w_ham + counter

    def _store_samples_in_memory(self):
        pickle.dump(self.dictionary, open(self.STOREDATA, 'wb'))
        pickle.dump(self.test_data, open(self.STORETESTDATA, 'wb'))

    def _is_memory(self):
        memory_name = Path(self.STOREDATA)
        return memory_name.is_file()

    def reset_memory(self):
        """
        Reset memory and re-train the agent
        """
        if self._is_memory():
            os.remove(self.STOREDATA)
            os.remove(self.STORETESTDATA)

