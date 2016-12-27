#!/usr/bin/python3

from __future__ import print_function
from nb_trainer import Trainer
from nb_classifier import Classifier


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train_agent()
    classifier = Classifier(trainer.offset)
    user_sentence = input("Write your sentence and check if it's ham or spam: ")
    classifier.classify(user_sentence)

    if classifier.is_spam():
        print("SPAM!")
    else:
        print("HAM!")

