import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

from dataset_builder import DatasetBuilder
from file_loader import f
from architectures import BinaryClassifier
from utils import dataset_split, make_train_step, accuracy, generate_datapoints, predict_all
from ground_truth import Truth
from validator import Validator

#ex = Experiment("hello_sacred")
ex = Experiment("simple_expert_embedding")

ex.observers.append(MongoObserver.create(url='localhost:27017',
                                        #db_name='hello_sacred'))
                                         db_name='simple_expert_embedding'))

data = {"queries": f.queries, "experts": f.experts, "skills": f.skills, "query2id": f.query2id, "expert2id": f.expert2id, "skill2id": f.skill2id}


@ex.config
def cfg():
    # hyperparameters regarding the dataset
    feature_names = ["country", "sex", "age range"]
    query = False
    wordemb_key = 'skill_docvec'

    # hyperparameters regarding the network architecture
    expemb_size = 50
    lin_layer_sizes = [100]
    expemb_dropout = 0.01
    lin_layer_dropouts = [0.1]

    # hyperparameters regarding the network training
    lr = 0.01
    epochs = 3
    batch_size = 34

@ex.main
def run(feature_names, query, wordemb_key, expemb_size, lin_layer_sizes, expemb_dropout, lin_layer_dropouts, lr, epochs, batch_size, id):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # TRAINING
    # build dataset, train/test split and data loaders
    dataset = DatasetBuilder(data, feature_names=feature_names, query=query, wordemb_key=wordemb_key)
    train_loader, validation_loader = dataset_split(dataset, validation_split=0.2, batch_size=batch_size, seed=42)

    # create model with hyperparameters and set criterion
    model = BinaryClassifier((dataset.expert_id_length, expemb_size), dataset.input_size, lin_layer_sizes=lin_layer_sizes,
               output_size=1, emb_dropout=expemb_dropout, lin_layer_dropouts=lin_layer_dropouts).to(device).double()
    criterion = nn.BCELoss()    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_step = make_train_step(model, criterion, optimizer)

    accuracies = []

    for epoch in range(epochs):
        for emb_x, feat_x, y in train_loader:


            emb_x = emb_x.to(device)
            feat_x = feat_x.to(device)
            y  = y.to(device)

            # Forward Pass
            # run training step as explained in the helper function
            preds, loss = train_step(emb_x, feat_x, y)
            ex.log_scalar("train_losses", loss)

            with torch.no_grad():
                ex.log_scalar("train_accuracies", accuracy(preds.cpu(), torch.tensor(y, dtype=torch.long).cpu()))

        # deactivate gradients to store test loss information for every epoch
        with torch.no_grad():
            for emb_x_val, feat_x_val, y_val in validation_loader:
                emb_x_val = emb_x_val.to(device)
                feat_x_val = feat_x_val.to(device)
                y_val = y_val.to(device)

                model.eval()

                preds = model(emb_x_val, feat_x_val)
                val_loss = criterion(preds, y_val)
                ex.log_scalar("val_losses", val_loss.item())
                accuracies.append(accuracy(preds.cpu(), torch.tensor(y_val, dtype=torch.long).cpu()))
                ex.log_scalar("val_accuracies", accuracy(preds.cpu(), torch.tensor(y_val, dtype=torch.long).cpu()))

    # save model
    torch.save(model.state_dict(), './models/' + str(id) + '.tar')

    # EVALUATION
    '''datapoints = generate_datapoints(dataset)
    dataset = DatasetBuilder(data, feature_names, datapoints = datapoints,query=query, wordemb_key=wordemb_key)
    rankings = predict_all(dataset, model, device)

    truth = Truth(f.document2expert, data, with_query=query, wordemb_key=wordemb_key, top_k = 10)
    validator = Validator()
    validator.validate(rankings, truth)
    
    ex.log_scalar("mean_ndcg", np.mean(validator.ndcg))
    ex.log_scalar("mean_precisionk", np.mean(validator.precisionk))'''

    return np.mean(accuracies[-batch_size:])