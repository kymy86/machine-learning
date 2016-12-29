#!/usr/bin/python3

from __future__ import print_function
from nb_trainer import Trainer
from nb_classifier import Classifier


if __name__ == '__main__':
    print("I'm training the agent......")
    trainer = Trainer(0.85)
    trainer.train_agent()
    print("The training is over\n")
    classifier = Classifier(trainer.offset, Trainer.STOREDATA, Trainer.STORETESTDATA)
    print("Executing agent on the test data....\n\n")
    tot_spam = 0
    spam = 0
    accuracy = 0
    error = 0

    for doc in classifier.test_data:
        if doc[0] == 'spam':
            tot_spam += 1

    print("Tot size dataset {}".format(len(classifier.test_data)))
    print("Tot real spam {}".format(tot_spam))

    for doc in classifier.test_data:
        classifier.treshold = 0
        classifier.classify(doc[1])
        if classifier.is_spam():
            spam += 1
        if classifier.is_spam() and doc[0] == 'spam':
            accuracy += 1
        if classifier.is_spam() and doc[0] == 'ham':
            error += 1

    print("Real SPAM detected {}".format(accuracy))
    print("Tot spam recognized {}".format(spam))
    print("SPAM accuracy {}%".format(float(accuracy/tot_spam)*100))
    print("False positive {}%".format(float(error/tot_spam)*100))


