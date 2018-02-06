#!/usr/bin/env python3

import classifier_tools as tools
from sentiment_values import INDETERMINATE_SENTIMENT

import csv
import itertools
from ast import literal_eval
import nltk
from sklearn.svm import SVC
import numpy as np
import pickle



######################################################################
def tokenizeTweets(tweet_collection):
    toknizer = tools.getTokenizer()
    stop_tokens = tools.getStopTokens()
    tokenized_tweets = []
    for tweet_observation in tweet_collection:
        target_class = literal_eval(tweet_observation[1])
        if (target_class != INDETERMINATE_SENTIMENT):
            tokens = toknizer.tokenize(tools.replaceQuotations(tweet_observation[0]))
            tokenized_tweets.append((list(filter(lambda word: word not in stop_tokens, tokens)), target_class))
    return tokenized_tweets
    

def pickleResource(resource, path):
    f = open(path + ".pickle", 'wb')
    pickle.dump(resource, f)
    f.close()


def main():
    # Collection of Trump tweets classfied as postive, negative, or indeterminate
    with open("TrumpTweets2.csv") as trump_tweets_file:
        trump_tweets_dataset = itertools.islice(csv.reader(trump_tweets_file), 1, 805)
    
        # Construct dataset with two columns: tokenized document and target class
        tokenized_tweet_dataset = tokenizeTweets(trump_tweets_dataset)
    
    # Construct training data for use by nltk and sklearn classifiers
    lexicon = tools.createLexicon(tools.getDatasetDocuments(tokenized_tweet_dataset))
    nltk_featuresets = [(tools.createDocumentFeaturesForNltk(d,lexicon), c) for (d,c) in tokenized_tweet_dataset]
    sklearn_featuresets = np.array([list(tools.createDocumentFeaturesForSklearn(d,lexicon)) for (d,c) in tokenized_tweet_dataset])
    target_classes = np.array([c for (d,c) in tokenized_tweet_dataset])

    # Train classifiers
    trump_tweets_classifiers = []
    trump_tweets_classifiers.append((nltk.MaxentClassifier.train(nltk_featuresets, nltk.classify.MaxentClassifier.ALGORITHMS[0], max_iter=10), "maxent"))
    trump_tweets_classifiers.append((nltk.NaiveBayesClassifier.train(nltk_featuresets), "naive_bayes"))
    trump_tweets_classifiers.append((SVC(kernel='linear', probability=True).fit(sklearn_featuresets, target_classes), "linear_svm"))
    trump_tweets_classifiers.append((SVC(gamma=0.1).fit(sklearn_featuresets, target_classes), "rbf01_svm"))
    trump_tweets_classifiers.append((SVC(gamma=0.05).fit(sklearn_featuresets, target_classes), "rbf005_svm"))
    trump_tweets_classifiers.append((SVC(gamma=0.2).fit(sklearn_featuresets, target_classes), "rbf02_svm"))
    
    # Save classifiers
    for classifier, name in trump_tweets_classifiers:
        pickleResource(classifier, "classifiers/" + name + "trump_tweets_classifier")
    
    # Save featuresets
    pickleResource(nltk_featuresets, "nltk_featuresets")
    pickleResource(sklearn_featuresets, "sklearn_featuresets")
    pickleResource(target_classes, "target_classes")


if __name__ == '__main__':
    main()
