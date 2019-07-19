import torch
from torch.utils.data import Dataset
import numpy as np

class DatasetBuilder(Dataset):
    def __init__(self, data, feature_names, datapoints = None, wordemb_key="query_docvec", with_nonnatural = False):
        """
        Characterizes a Dataset for PyTorch

        Parameters
        ----------

        data: dict, Dictionary with all relevant dictionaries (because it looks nice) 
        feature_names: Choose the features you want to integrate in your input ["sex","country",etc.]
        datapoints: You may as well define your datapoints
        wordemb_key: The key name of the word vectors you want to choose ["query_docvec","query_noStopwords_docvec","skill_docvec","skill_ft_meanvec","skill_bpft_docvec","skill_bp_docvec"]
        """
        
        # decides whether to use queries or skills dictionary to get the word vectors
        if wordemb_key in ["query_docvec","query_noStopwords_docvec"]:
            self.wordembs = data["queries"]
            self.wordemb2id = data["query2id"]
        else: 
            self.wordembs = data["skills"]
            self.wordemb2id = data["skill2id"]
    
        self.experts = data["experts"]
        self.expert_id_length = len(set(self.experts))
        
        self.expert2id = data["expert2id"]
        
        if datapoints == None:
            self.datapoints = []
            # create list of all datapoints á la [(expert ID, query/skill ID, score), ...]
            for idx, expert in self.experts.items():
                if wordemb_key in ["query_docvec","query_noStopwords_docvec"]:
                    for element in expert['queries']:
                        # filter out scores of 1
                        if element[1] != 1:
                            self.datapoints.append((idx, element[0], element[1]))
                # else clause because the skills dict has a slightly different shape [[3,75],2] 
                # instead of [75, 2] per element
                else: 
                    for element in expert['skills']:
                        # filter out scores of 1
                        if element[1] != 1:
                            if with_nonnatural == True:
                                self.datapoints.append((idx, element[0][0], element[1]))
                                if len(element[0]) > 1:
                                    self.datapoints.append((idx, element[0][1], element[1]))
                            else:
                                if self.wordembs[element[0][0]]['non-natural'] != 1:
                                    self.datapoints.append((idx, element[0][0], element[1]))
                                if len(element[0]) > 1:
                                    if self.wordembs[element[0][1]]['non-natural'] != 1:
                                        self.datapoints.append((idx, element[0][1], element[1]))
        
        else:
            self.datapoints = datapoints
        
        input_size = 0
        # identify the length of the one hot encoded categorical feature vectors and their mapping
        # collect features from the input dicts
        feature_lists = [[] for name in feature_names]
        for key,value in self.experts.items():
            for idx, feature in enumerate(feature_names): 
                feature_lists[idx].append(value[feature])
        # get their unique values and create dict for mapping int's to categories
        feature_uniques = [{} for name in feature_names]
        for mdx, feature in enumerate(feature_lists):
            for idx, element in enumerate(sorted(list(set(feature)))):
                feature_uniques[mdx][element] = idx
                input_size += 1
        
        # length of the word embedding used, e.g. 4396 in flair or 300 in fasttext
        wordemb_size = self.wordembs[0][wordemb_key].shape[0]
        # total length of the entire input array without expert embedding
        input_size += wordemb_size
        
        # make sure data is shuffled before it's broken down into the 
        # three matrices "expert_id", "feature_input" and "label"
        
        expert_id = np.zeros((len(self.datapoints)))
        label = np.zeros((len(self.datapoints),1))
        input_matrix = np.zeros((len(self.datapoints), input_size))
        # iterate through datapoints and build features right into the input matrix
        for mdx, datapoint in enumerate(self.datapoints):
            row_dancer = 0
            expert_id[mdx] = datapoint[0]
            # give '0' scores a 0 label and '2' or '3' scores a 1 label
            if datapoint[2] == 0:
                label[mdx, 0] = datapoint[2]
            else: 
                label[mdx, 0] = 1
            # break down categorical feature and encode one hot into input matrix
            for idx, feature in enumerate(feature_names):
                feature_position = feature_uniques[idx][self.experts[datapoint[0]][feature]]
                input_matrix[mdx ,feature_position + row_dancer] = 1
                row_dancer += len(feature_uniques[idx])
            # retrieve word vector for query/skill and inject into input matrix
            input_matrix[mdx, input_size-wordemb_size:] = self.wordembs[datapoint[1]][wordemb_key]
            
        self.input_size = input_size
        self.expert_id = torch.tensor(expert_id, dtype=torch.long)
        
        self.input_matrix =  torch.tensor(input_matrix).type('torch.DoubleTensor')
        self.label = torch.tensor(label)
                
    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return len(self.datapoints)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.expert_id[idx], self.input_matrix[idx], self.label[idx]]