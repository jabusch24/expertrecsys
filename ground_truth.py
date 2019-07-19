import numpy as np

from utils import get_dcg

class Truth:
    '''This class holds the relevance score matrix, preprocessed and ready to compute.'''
    def __init__(self, document2expert, data, with_query=True, wordemb_key='query_noStopwords_docvec', top_k = 10):
        self.document2expert = document2expert
        self.wordemb_key = wordemb_key
        self.with_query = with_query
        self.top_k = top_k
        # numpy matrix with queries as rows and expert relevance scores as columns
        self.expert2query = []
        self.expert2querynan = []
        self.idcg = []
        self.expert2id = data['expert2id']
        self.experts = data['experts']
        self.skill2id = data['skill2id']
        self.skills = data['skills']
        self.queries = data['queries']
        self.query2id = data['query2id']
        
        # zero matrix for queries as rows and experts as columns
        if with_query:
            expert2query = np.zeros((len(self.query2id), len(self.expert2id)))
            expert2querynan = np.zeros((len(self.query2id), len(self.expert2id)))
        else:
            expert2query = np.zeros((len(self.skill2id), len(self.expert2id)))
            expert2querynan = np.zeros((len(self.skill2id), len(self.expert2id)))
        expert2querynan[:] = np.nan
        
        #expert2query[:] = np.nan
        for key, value in self.experts.items():
            if with_query:
                for element in value['queries']:
                    expert2query[element[0], key] = element[1]
                    expert2querynan[element[0], key] = element[1]
            else:
                for element in value['skills']:
                    expert2query[element[0], key] = element[1]
                    expert2querynan[element[0], key] = element[1]

        self.expert2query = expert2query
        self.expert2querynan = expert2querynan
        self.__get_idcg()
        
    def __get_idcg(self):
        for row in np.flip(np.sort(self.expert2query)):
            self.idcg.append(get_dcg(row))