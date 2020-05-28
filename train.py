from sklearn.model_selection import ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import time
import utils
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds, eigs
import os
import joblib
#import handle_data as H
#

#Perform Train Test Split on the data. 
def splitData(df, test_size):
    if test_size == 0:
        train_idx = np.array(df.index)
        valid_idx = np.array([])
    else:
        splitter = ShuffleSplit(n_splits=1, random_state=6, test_size=test_size, train_size=None)
        train_test_idx = list(splitter.split(df))
        train_idx = train_test_idx[0][0]
        valid_idx = train_test_idx[0][1]
    return train_idx, valid_idx

def count(df,n_gram, lower_word_threshold=0.001, upper_word_threshold=0.6):
    # Count Vectorizer for counting.
    ngram = tuple(n_gram)
    countVec = CountVectorizer(min_df=lower_word_threshold, max_df=upper_word_threshold,ngram_range = ngram)
    countMatrix = countVec.fit_transform(df['nlp_text'])
    isCounted = True
    return countVec, countMatrix

def fitLDA(countMatrixTrain,n_topic,random_state,alpha=None):
    '''
    Perform Latent Dirichlet Allocation to cluster documents, and create topic distribution features.
    Call count method prior to this method
    '''
    if alpha is None: # If topic prior is not given, assign it as 1/k
        alpha = 1/n_topic
    print(f'Latent Dirichlet Allocation has been started for overall Corpus.')
    model = LatentDirichletAllocation(n_components=n_topic, random_state=random_state,doc_topic_prior=alpha, learning_method='online',batch_size=2048,verbose=1, n_jobs=-2)
    # Train the model
    t = time.time() # tic
    model.fit(countMatrixTrain)
    elapsed = time.time() - t # toc
    print(f'LDA is fit. Elapsed time: in {elapsed} seconds.')
    t = time.time()  # tic
    # Transform all data into topics
    modelcomponents = model.components_
    beta_lda = modelcomponents/modelcomponents.sum(axis=1)[:, np.newaxis]
    return model,beta_lda

def transformer(countMatrix, model):
    '''
    Using count matrix, evaluate topic distributions of all documents.
    '''
    topic_dist = model.transform(countMatrix)
    dominant_topic = np.argmax(topic_dist,axis=1)
    return topic_dist, dominant_topic

def getEmbedding(paragraph, model, countVec):

    '''This function transforms raw text into topic distribution. In other words, Get Embedding of a paragraph, and its sentences.'''
    sents = H.docprocess(paragraph, True)
    sent_emb = countVec.transform(sents)
    #
    sent_vec = model.transform(sent_emb)
    return sent_vec


def saveModel(modelname,datafolder,model,countVec,countMatrix,n_topic):
    '''Save topic distribution if desired.'''
    print('Writing to Binary File on disk...')
    filename = os.path.join(datafolder,'models',f'{modelname}.joblib')
    joblib.dump([model, countVec, countMatrix, n_topic],filename)
    print(f'Saved to file: {filename}\n')
    return 0

def recoverModel(modelname,datafolder):
    '''Recover the model.'''
    # model_,countVec_,countMatrix_,n_topic_ = recoverModel('mymodel30top',H.datafolder)
    filename = os.path.join(datafolder,'models',f'{modelname}.joblib')
    recoveredModel = joblib.load(filename)
    model = recoveredModel[0]
    countVec = recoveredModel[1]
    countMatrix = recoveredModel[2]
    n_topic = recoveredModel[3]
    beta_lda = model.components_/model.components_.sum(axis=1)[:, np.newaxis]
    return model,countVec,countMatrix,n_topic
