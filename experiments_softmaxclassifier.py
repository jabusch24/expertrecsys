import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

from dataset_builder import DatasetBuilder
from file_loader import f
from utils import dataset_split

#ex = Experiment("hello_sacred")
ex = Experiment("softmax_classifier")

ex.observers.append(MongoObserver.create(url='localhost:27017',
                                        #db_name='hello_sacred'))
                                         db_name='softmax_classifier'))

data = {"queries": f.queries, "experts": f.experts, "skills": f.skills, "query2id": f.query2id, "expert2id": f.expert2id, "skill2id": f.skill2id}


@ex.config
def cfg():
    # hyperparameters regarding the dataset
    feature_names = []
    wordemb_key = 'skill_docvec'

    # hyperparameters regarding the network architecture
    emb_dims = 50
    activations = 'sigmoid'
    
    # hyperparameters regarding the network training
    lr = 0.01
    epochs = 20
    batch_size = 1

@ex.main
def run(feature_names, wordemb_key, with_nonnatural, emb_dim, activation, lr, epochs, batch_size, criterion, id):

    # prepare data
    dataset = DatasetBuilder(data, feature_names=feature_names, wordemb_key=wordemb_key, with_nonnatural=with_nonnatural)
    train_loader, validation_loader = dataset_split(dataset, validation_split=0.2, batch_size=batch_size, seed=42)

    # set hyperparameters
    activation = activation
    emb_dim = emb_dim
    lr = lr
    input_size=dataset[0][1].shape[0]
    output_size=len(data["experts"])
    if criterion == "BCELoss":
        criterion = torch.nn.BCELoss()
    elif criterion == "BCEWithLogitsLoss":
        criterion = torch.nn.BCEWithLogitsLoss()

    # create network matrices
    W1 = Variable(torch.randn(emb_dim, input_size).float(), requires_grad=True)
    W2 = Variable(torch.randn(output_size, emb_dim).float(), requires_grad=True)

    # monitoring variable
    best_accuracy = 0


    # run model
    for epoch in range(epochs):
        # model monitoring variables
        loss_val = 0
        win = 0
        lose = 0
        for expert_id, input_emb, target in train_loader:
            # transform input data            
            x = Variable(input_emb).float().transpose(0,1)#.to(device)
            y_true = Variable(target, 0).float()#.to(device)
            expert_id = Variable(torch.unsqueeze(expert_id, 0).long())#.to(device)

            # run matrix multiplication
            z1 = torch.matmul(W1, x)
            a1 = torch.sigmoid(z1)
            z2 = torch.matmul(W2, a1)
            
            # run output activation
            if activation == 'sigmoid':
                a2 = torch.sigmoid(z2)
            elif activation == 'softmax':
                a2 = torch.softmax(z2, 0)
            else:
                a2 = z2
            
            # create loss and accuracy measures (if batch_size > 1 the output needs to be transformed)
            if batch_size > 1:
                a3 = torch.zeros((a2[0].shape[0],1))
                for idx, eid in enumerate(expert_id[0]):
                    a3[idx,0] = a2[eid, idx]
                loss = criterion(a3, y_true)
                
            else:
                loss = criterion(a2[expert_id][0], y_true)
        
            # backpropagation
            loss.backward()
            loss_val += loss
            
            W1.data -= lr * W1.grad.data
            W2.data -= lr * W2.grad.data
            
            # calculate accuracy measures
            if batch_size > 1:
                for result in torch.abs(a3 - y_true) < 0.5:
                    if result == 0:
                        lose += 1
                    else:
                        win += 1
            else:
                if torch.abs(a2[expert_id][0][0][0] - y_true[0][0]) < 0.5:
                    win += 1
                else:
                    lose += 1
                    
            # zero gradients
            W1.grad.data.zero_()
            W2.grad.data.zero_()
        ex.log_scalar("train_accuracies", win/(win+lose))
        ex.log_scalar("train_loss", int(loss_val))

        # deactivate gradients to store test loss information for every epoch
        with torch.no_grad():
            # model monitoring variables
            win = 0
            lose = 0
            # run model over validation dataset
            for expert_id, input_emb, target in validation_loader:
                # transform input data            
                x = Variable(input_emb).float().transpose(0,1)#.to(device)
                y_true = Variable(target, 0).float()#.to(device)
                expert_id = Variable(torch.unsqueeze(expert_id, 0).long())#.to(device)
                
                # run matrix multiplication
                z1 = torch.matmul(W1, x)
                a1 = torch.sigmoid(z1)
                z2 = torch.matmul(W2, a1)

                # run output activation
                if activation == 'sigmoid':
                    a2 = torch.sigmoid(z2)
                elif activation == 'softmax':
                    a2 = torch.softmax(z2, 0)
                else:
                    a2 = z2

                # calculate accuracy measures
                if batch_size > 1:
                    a3 = torch.zeros((a2[0].shape[0],1))
                    for idx, eid in enumerate(expert_id[0]):
                        a3[idx,0] = a2[eid, idx]
                    for result in torch.abs(a3 - y_true) < 0.5:
                        if result == 0:
                            lose += 1
                        else:
                            win += 1
                else:
                    if torch.abs(a2[expert_id][0][0][0] - y_true[0][0]) < 0.5:
                        win += 1
                    else:
                        lose += 1

            # log result        
            ex.log_scalar("val_accuracies", win/(win+lose))
            if win/(win+lose) > best_accuracy:
                best_accuracy = win/(win+lose) 

        if epoch % 5 == 0:    
            print(f'Loss at epo {epoch}: {int(loss_val)/len(dataset)}')

    # save model
    torch.save(W1, './models/softmax_classifier/softmax_classifier_W1_' + str(id) + '.tar')
    torch.save(W2, './models/softmax_classifier/softmax_classifier_W2_' + str(id) + '.tar')

    return best_accuracy