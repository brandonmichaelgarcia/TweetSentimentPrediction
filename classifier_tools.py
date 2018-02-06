#!/usr/bin/env python3

import nltk
import string


######################################################################
def getTokenizer():
    return nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)


def getStopTokens():
    meaningful_punctuation = '!?'
    stop_punctuation = list(x for x in string.punctuation if x not in meaningful_punctuation)
    more_stop_tokens = ["..."]
    return nltk.corpus.stopwords.words('english') + stop_punctuation + more_stop_tokens


def replaceQuotations(sentence):
    return sentence.replace('“','"').replace('”','"').replace('’', "'")


def getDatasetDocuments(dataset):
    return [elem[0] for elem in dataset]


def createLexicon(dataset_document):
    '''
    Gather all high frequency tokens for use in vectorizing documents
    '''
    text_tokens = [token for document in dataset_document for token in document]
    all_tokens = nltk.FreqDist(tok.lower() for tok in text_tokens)
    return [elem[0] for elem in all_tokens.most_common(2000)]


def createDocumentFeaturesForNltk(document, lexicon):
    '''
    Vectorizes a training document based on a provided lexicon (list of relevant tokens) 
    as needed by nltk's Maxent and Naive Bayes classifiers
    '''
    document_tokens = set(document) 
    return {tok:True for tok in lexicon if tok in document_tokens}


def createDocumentFeaturesForSklearn(document, lexicon):
    '''
    Vectorizes a training document based on a provided lexicon (list of relevant tokens) 
    as needed by sklearn's support vector classifier (SVC)
    '''
    document_tokens = set(document) 
    return [1 if tok in document_tokens else 0 for tok in lexicon]


def createFeatureSetFromTextForNltk(text):
    '''
    Tokenizes and vectorizes tokens for performing prediction on a small text sample
    as needed by nltk's Maxent and Naive Bayes classifiers
    '''
    tokens = getTokenizer().tokenize(replaceQuotations(text))
    stop_tokens = getStopTokens()
    filtered_tokens = list(filter(lambda word: word not in stop_tokens, tokens))
    return createDocumentFeaturesForNltk(filtered_tokens, filtered_tokens)


def createFeatureSetFromTextForSklearn(text):
    '''
    Tokenizes and vectorizes tokens for performing prediction on a small text sample
    as needed by sklearn's support vector classifier (SVC)
    '''
    tokens = getTokenizer().tokenize(replaceQuotations(text))
    stop_tokens = getStopTokens()
    filtered_tokens = list(filter(lambda word: word not in stop_tokens, tokens))
    return createDocumentFeaturesForSklearn(filtered_tokens, filtered_tokens)



    