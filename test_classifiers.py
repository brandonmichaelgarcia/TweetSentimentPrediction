#!/usr/bin/env python3


import nltk
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pickle


######################################################################
CLASSIFIERS = ['maxent', 'naive_bayes', 'linear_svm', 'rbf01_svm', 'rbf02_svm']

def getROC(predictions, target_classes):
    tp, fp, tn, fn = [0] * 4
    for prediction,target_class in zip(predictions, target_classes):
        if (prediction == 1):
            if (target_class == 1):
                tp += 1
            else:
                fp += 1
        else:
            if (target_class < 1):
                tn += 1
            else:
                fn += 1
             
    tpr = tp/(tp + fn)
    fpr = fp/(fp + tn)
    acc = (tp + tn)/(tp + tn + fp + fn)
    return tpr, fpr, acc
    
    
def main():
    with open('nltk_featuresets.pickle', 'rb') as f:
        nltk_featuresets = pickle.load(f)
    with open('sklearn_featuresets.pickle', 'rb') as f:
        sklearn_featuresets = pickle.load(f)
    with open('target_classes.pickle', 'rb') as f:
        target_classes = pickle.load(f)
        
        
    trump_tweets_classifiers = []
    trump_tweets_classifiers.append((nltk.MaxentClassifier, "maxent"))
    trump_tweets_classifiers.append((nltk.NaiveBayesClassifier, "naive_bayes"))
    trump_tweets_classifiers.append((SVC(kernel='linear', probability=True), "linear_svm"))
    trump_tweets_classifiers.append((SVC(gamma=0.1), "rbf01_svm"))
    trump_tweets_classifiers.append((SVC(gamma=0.05), "rbf005_svm"))
    trump_tweets_classifiers.append((SVC(gamma=0.2), "rbf02_svm"))    

    classifier_tpr = []
    classifier_fpr = []
    classifier_acc = []
    num_splits = 5
    cv = StratifiedKFold(n_splits=num_splits)
    for train, test in cv.split(nltk_featuresets, target_classes):
        training_set = [nltk_featuresets[i] for i in list(train)]
        testing_set = [nltk_featuresets[i] for i in list(test)]
        split_tpr = []
        split_fpr = []
        split_acc = []
        for classifier, name in trump_tweets_classifiers:
            if name == "maxent":
                trained_classifier = classifier.train(training_set, nltk.classify.MaxentClassifier.ALGORITHMS[0], max_iter=10)
            elif name == "naive_bayes":
                trained_classifier =  nltk.NaiveBayesClassifier.train(training_set)
            else:
                trained_classifier = classifier.fit(sklearn_featuresets[train], target_classes[train])
            
            if name == "maxent" or name == "naive_bayes":
                predictions = [trained_classifier.classify(nltk_featuresets[i][0]) for i in list(test)]
            else:
                predictions = trained_classifier.predict(sklearn_featuresets[test])
            
            tpr, fpr, acc = getROC(target_classes[test], predictions[:])
            split_tpr.append(tpr)
            split_fpr.append(fpr)
            split_acc.append(acc)
        classifier_tpr.append(split_tpr)
        classifier_fpr.append(split_fpr)
        classifier_acc.append(split_acc)


    for i,name in enumerate(CLASSIFIERS):
        print(name + " tpr: = {}".format(np.mean([row[i] for row in classifier_tpr])))
        print(name + " fpr: = {}".format(np.mean([row[i] for row in classifier_fpr])))
        print(name + " acc: = {}".format(np.mean([row[i] for row in classifier_acc])))
    
    
if __name__ == '__main__':
    main()

    