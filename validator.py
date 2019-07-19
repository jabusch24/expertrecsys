import numpy as np

from utils import get_dcg, get_precisionk

class Validator:
    '''This class holds the search function and the results of the search function 
    with the queries from the Queries class and the relevance scores from the Validator class.'''
    def __init__(self):
        self.dcg = []
        self.ndcg = []
        self.precisionk = []
        # result of the search function in document ID, list of numpy arrays with document IDs in order of relevance
        self.results = []
        # search function results mapped to experts (first mentioned expert first) [['Alexander Mundt', 'Tatjana Gust']]
        self.rankings = []
        # search function results mapped to relevance scores, list of numpy arrays with top_k relevance scores
        self.scores = []
        self.scoresnan = []
                
    def __get_scores(self, expert2id, expert2query, expert2querynan, top_k):
        scores = []
        scoresnan = []
        for idx, result in enumerate(self.rankings):
            expert_ids = [expert2id[expert] for expert in result[0:top_k]]
            scores.append(expert2query[idx, expert_ids])
            scoresnan.append(expert2querynan[idx, expert_ids])
        self.scores = scores
        self.scoresnan = scoresnan
    
    def __get_ndcg(self, idcg):
        self.dcg = [get_dcg(score) for score in self.scores]
        self.ndcg = np.divide(self.dcg, idcg)
        
    def __get_precisionk(self):
        self.precisionk = [get_precisionk(score) for score in self.scores]

    def validate(self, rankings, truth):
        self.rankings = rankings
        self.__get_scores(truth.expert2id, truth.expert2query, truth.expert2querynan, truth.top_k)
        self.__get_ndcg(truth.idcg)
        self.__get_precisionk()
        return self.ndcg