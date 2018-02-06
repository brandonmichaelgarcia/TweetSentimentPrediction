#!/usr/bin/env python3

import classifier_tools as tools
from sentiment_values import POSITIVE_SENTIMENT

import nltk
import pickle
import socket
import sys

######################################################################
class SentimentClassifiers(object):
    def __init__(self):
        #self.prediction = 
        with open('/classifiers/maxent_trump_tweets_classifier.pickle','rb') as f:
            self.bayes_classifier = pickle.load(f)
        with open('/classifiers/naive_bayes_trump_tweets_classifier.pickle','rb') as f:
            self.maxent_classifier = pickle.load(f) 
        # with open('linear_svm_classifier_trump_tweets.pickle','rb') as f:
        #     self.linear_classifier = pickle.load(f) 
        # with open('rbf01_svm_classifier_trump_tweets.pickle','rb') as f:
        #     self.rbf01_classifier = pickle.load(f) 
        # with open('rbf02_svm_classifier_trump_tweets.pickle','rb') as f:
        #     self.rbf02_classifier = pickle.load(f) 

    def getSentiment(self,input_data):
        text = input_data
        print(text)
        
        maxent_result = int(self.maxent_classifier.classify(tools.createFeatureSetFromTextForNltk(text)))
        bayes_result = int(self.bayes_classifier.classify(tools.createFeatureSetFromTextForNltk(text)))
        # linear_result = int(self.linear_classifier.predict(CreateFeatureSetFromTextForSVM(sentence)))
        # rbf01_result = int(self.rbf01_classifier.predict(CreateFeatureSetFromTextForSVM(sentence)))
        # rbf02_result = int(self.rbf02_classifier.predict(CreateFeatureSetFromTextForSVM(sentence)))

        print("Bayes Result = {}".format(bayes_result))
        print("Maxent Result = {}".format(maxent_result))
        # print("linear Result = {}".format(linear_result))
        # print("rbf01 Result = {}".format(rbf01_result))
        # print("rbf02 Result = {}".format(rbf02_result))
        
        # if (maxent_result + bayes_result + linear_result + rbf01_result + rbf02_result >= 1):
        #     classifier_assemblage = 1
        # else:
        #     classifier_assemblage = -1
        return bayes_result #classifier_assemblage


def main():
    classifiers = SentimentClassifiers()
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 10000)
    print('starting up on %s port %s' % server_address)
    sock.bind(server_address)
    
    sock.listen(1)
    while True:
        print('waiting for a connection')
        connection, client_address = sock.accept()
        try:
            print('connection from {}'.format(client_address[0]))
            while True:
                data = connection.recv(1024)  
                  
                if not data: break
                
                sentiment = int(classifiers.getSentiment(str(data,'utf-8')))
                sentiment_msg = b'Positive' if sentiment == POSITIVE_SENTIMENT else b'Negative'
                connection.sendall(sentiment_msg)
            
        finally:
            connection.close()


if __name__ == '__main__':
    main()