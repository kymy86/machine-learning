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
    false_positive = 0
    false_negative = 0

    for doc in classifier.test_data:
        if doc[0] == 'spam':
            tot_spam += 1

    print(f"Dataset size {len(classifier.test_data)}")
    print(f"Spam in dataset {tot_spam}")

    for doc in classifier.test_data:
        classifier.treshold = 0
        classifier.classify(doc[1])
        if classifier.is_spam() and doc[0] == 'spam':
            #True Negatives
            accuracy += 1
        if classifier.is_spam() and doc[0] == 'ham':
            #False positive: how many hams are classified as spam
            false_positive += 1
        if not classifier.is_spam() and doc[0] == 'spam':
            #False Negative: how many spams aren't classified as spam
            false_negative += 1

    print(f"True Negatives {accuracy}")
    print(f"False Negative {false_negative}")
    print(f"False Positive {false_positive}")
