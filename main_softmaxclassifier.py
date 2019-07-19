# omniboard -m localhost:27017:hello_sacred
from experiments_softmaxclassifier import ex
import numpy as np


def do():
    # hyperparameters regarding the dataset
    id = 0
    feature_names = []
    query_combinations = ['query_docvec', 'query_noStopwords_docvec', 'skill_bp_docvec', 'skill_docvec','skill_ft_meanvec','skill_bpft_docvec']
    with_nonnaturals = [True, False] 

    # hyperparameters regarding the network architecture
    emb_dims = [50, 100, 200, 500, 1000]
    activations = [None, 'sigmoid', 'softmax']


    # hyperparameters regarding the network training
    lrs = [0.001, 0.01, 0.1]
    epochs = 20
    batch_sizes = [1,5,20]
    criterions = ['BCELoss', 'BCEWithLogitsLoss']

    for query_combination in query_combinations:
        for emb_dim in emb_dims:
            for lr in lrs:
                for batch_size in batch_sizes:
                    for activation in activations:
                        for criterion in criterions:
                            if criterion == 'BCELoss' and activation == None:
                                pass
                            else:
                                config_updates={'feature_names':feature_names,
                                                'wordemb_key':query_combination,
                                                'with_nonnatural': with_nonnaturals[np.random.randint(2)],
                                                'emb_dim': emb_dim,
                                                'activation': activation,
                                                'lr': lr,
                                                'epochs': epochs,
                                                'batch_size': batch_size,
                                                'criterion' : criterion,
                                                'id': id}

                                ex.run(config_updates=config_updates)
                                id += 1

if __name__ == '__main__':
    do()
