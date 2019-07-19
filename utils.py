import numpy as np
from numpy.linalg import norm
from gensim.utils import simple_preprocess
from collections import Counter
import pandas as pd

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from file_loader import f

def generate_datapoints(dataset):
    # create dataset containing all possible queries
    max_expert_id = torch.max(dataset.expert_id)
    max_query_id = len(dataset.wordemb2id)
    pred_datapoints = []
    for idx in range(0, max_expert_id + 1):
        for qid in range(0, max_query_id):
            pred_datapoints.append((idx, qid, 1))
    return pred_datapoints

def get_dcg(scores):
    dcg = 0
    # iterate through the ranks
    for idx, num in enumerate(scores):
        temp = np.power(2, num)/np.log((idx+1) + 1)
        dcg += temp
    return dcg

def get_precisionk(scores):
    irrelevant = sum([1 for score in scores if score == 0])
    relevant = len(scores) - irrelevant
    if relevant == 0 or irrelevant == 0:
        return relevant * 1
    else:
        return relevant/len(scores)

def predict_all(dataset, model, device):
    # make predictions with new dataset
    preds = np.zeros(len(dataset))
    for i in range(0, int(np.ceil(len(dataset)/500))):
        preds[i*500:(i+1)*500] = model(dataset[i*500:(i+1)*500][0].to(device), dataset[i*500:(i+1)*500][1].to(device)).detach().cpu().numpy()[:,0]
    
    # turn predictions into an expert/query matrix
    predictions = np.zeros((len(dataset.expert2id),len(dataset.wordemb2id)))
    for idx, point in enumerate(dataset.datapoints):
        predictions[point[0],point[1]] = preds[idx]

    simple_rankings = []
    for query in predictions.transpose():
        simple_rankings.append([f.experts[value]["expert"] for value in np.flip(np.argsort(query))])

    return simple_rankings

def dataset_split(dataset, validation_split=0.2, batch_size=34, seed=42):
    # prepare data for training
    validation_split = validation_split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    torch.manual_seed(seed)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader

# helper function to run training steps in a consolidated fashion
def make_train_step(model, criterion, optimizer):
    # train step is essentially the common forward + backprop path bundled together for ease of use
    def train_step(emb_x, feat_x, y):
        model.train()
        preds = model(emb_x, feat_x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return preds, loss.item()
    
    return train_step

def torch_accuracy(preds, y):
    y_hat = preds.clone().detach().numpy()
    y_true = y.clone().detach().numpy()
    true_values = np.abs(np.squeeze(y_hat)-np.squeeze(y_true))<0.5
    if true_values.shape == ():
        true_values = np.expand_dims(true_values,0)
    bin_count = np.bincount(true_values)
    if bin_count.shape == (1,):
        bin_count = np.append(bin_count, 0)
    return bin_count[1]/np.sum(bin_count)


# input are 2 numpy matrices
def cosSim(query, skills):
    try:
        query.shape[1]
    except:
        query = np.expand_dims(query, axis=0)
    try:
        skills.shape[1]
    except:
        skills = np.expand_dims(skills, axis=0)
    return np.nan_to_num(np.dot(query, skills.transpose())/np.dot(np.matrix(norm(query, axis=1)).transpose(), np.matrix(norm(skills, axis=1))))

flatten = lambda l: [item for sublist in l for item in sublist]

def build_tf(documents):
     # dictionary holding the final tf idf scores
    tf = {}

    for did, value in documents.items():
        # tokenize document
        tokens = simple_preprocess(value)
        # store number of tokens
        num_tokens = len(tokens)
        # create dataframe of unique tokens and their counts
        unique_tokens = pd.DataFrame(dict(Counter(tokens)).items(), columns=["word","count"])
        # calculate tf
        unique_tokens["tf"] = unique_tokens["count"]/num_tokens

        # go through each unique token and add it to tf
        for row in unique_tokens.iterrows():
            # if token already exists in tf, the try block will succeed
            try:
                tf[row[1][0]] = np.concatenate((tf[row[1][0]], np.array([[did, row[1][2]]])))
            # otherwise the token will be created in tf
            except:
                tf[row[1][0]] = np.array([[did, row[1][2]]])
    
    return tf

def build_idf(tf, num_documents):
     # build idf scores after all documents have been analyzed
    idf = {}
    # go through each word
    for key, value in tf.items():
        idf[key] = np.log(num_documents/value.shape[0])
    return idf

def build_tfidf(tf, idf):
    # calculate tfidf score for each word in each document
    for key, value in tf.items():
        tfidf = np.array([value[:,1]*idf[key]]).transpose()
        # add tfidf score to the numpy matrix of the term frequency
        tf[key] = np.concatenate((value, tfidf), axis=1)
        
    return tf

class TfidfIndex:
    '''Class holding the tfidf index and providing ability to search the index.'''
    def __init__(self):
        self.__num_documents = 0
        self.__tf = {}
        self.__idf = {}
        self.index = {}

    def build(self, documents):
        self.__num_documents = len(documents)
        self.__tf = build_tf(documents)
        self.__idx = build_idf(self.__tf, self.__num_documents)
        self.__index = build_tfidf(self.__tf, self.__idx)

    def search(self, search_term):
        '''Return list of document indices, sorted by relevance to the input query.'''
        try:
            indices = np.argsort(self.__index[search_term][:,2])[::-1]
            return self.__index[search_term][indices,0]
        except KeyError as e:
            print(e ,"is not part of our vocabulary.")
            print(e)
      