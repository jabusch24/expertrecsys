# omniboard -m localhost:27017:hello_sacred
from experiments_binaryclassifier import ex


def do():
    # hyperparameters regarding the dataset
    id = 0
    feature_names = []
    query_combinations = [(True, 'query_docvec'),(True, 'query_noStopwords_docvec'),(False, 'skill_bp_docvec'),(False, 'skill_docvec'),(False,'skill_ft_meanvec'),(False,'skill_bpft_docvec')] 

    # hyperparameters regarding the network architecture
    expemb_sizes = [50, 100, 200]
    lin_layer_sizess = [[1000,500],[500,500],[1000],[500],[]]
    expemb_dropout = 0.01
    lin_layer_dropoutss = [[0.1,0.05], [0.1,0.05], [0.1],[0.1],[]]

    # hyperparameters regarding the network training
    lrs = [0.001, 0.01]
    epochs = 3
    batch_size = 34

    for expemb_size in expemb_sizes:
        for ddx, lin_layer_sizes in enumerate(lin_layer_sizess):
            for combination in query_combinations:
                for lr in lrs:
                    config_updates={'feature_names':feature_names,
                                    'query':combination[0],
                                    'wordemb_key':combination[1],
                                    'expemb_size':expemb_size,
                                    'lin_layer_sizes': lin_layer_sizes,
                                    'lin_layer_dropouts': lin_layer_dropoutss[ddx],
                                    'lr': lr,
                                    'epochs': epochs,
                                    'batch_size': batch_size,
                                    'id': id}

                    ex.run(config_updates=config_updates)
                    id += 1

if __name__ == '__main__':
    do()
