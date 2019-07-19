import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifier(nn.Module):
    def __init__(self, emb_dim, feature_size, lin_layer_sizes=[500,500],
               output_size=1, emb_dropout=0.01, lin_layer_dropouts=[0.1,0.05]):
        """
        Parameters
        ----------

        emb_dim: Tuple, first element stating the number of experts 
            and second number to indicate embedding size

        lin_layer_sizes: List of integers.
          The size of each linear layer. The length will be equal
          to the total number
          of linear layers in the network.

        output_size: Integer
          The size of the final output.

        emb_dropout: Float
          The dropout to be used after the embedding layers.

        lin_layer_dropouts: List of floats
          The dropouts to be used after each linear layer.
        """
        super().__init__()

        # Embedding layers
        self.emb_layer = nn.Embedding(emb_dim[0],emb_dim[1])

        self.emb_size = emb_dim
        self.feature_size = feature_size
        
        # Linear Layers
        first_lin_layer = nn.Linear(self.emb_size[1] + self.feature_size, lin_layer_sizes[0])

        self.lin_layers =  nn.ModuleList([first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) for i in range(len(lin_layer_sizes) - 1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.feature_size)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])
        
    def forward(self, expert_id, input_matrix):


        x = self.emb_layer(expert_id) 
        x = self.emb_dropout_layer(x)


        normalized_cont_data = self.first_bn_layer(input_matrix)

        x = torch.cat([x, normalized_cont_data], 1) 

        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.droput_layers, self.bn_layers):

            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = torch.sigmoid(self.output_layer(x))

        return x


class SoftmaxClassifier(nn.Module):
    def __init__(self, input_size, emb_dim, output_size, activation=None):
        """
        Parameters
        ----------

        emb_dim: Tuple, first element stating the number of experts 
            and second number to indicate embedding 
            
        input_size: Dimension of the input layer

        output_size: Integer
          The size of the final output.
        """
        super().__init__()

        # Embedding layers
#         self.h1 = Variable(torch.randn(emb_dim, input_size).float(), requires_grad=True)
#         self.h2 = Variable(torch.randn(output_size, emb_dim).float(), requires_grad=True)  
        self.h1 = nn.Linear(input_size, emb_dim)
        self.h2 = nn.Linear(emb_dim, output_size)
        
        self.emb_size = emb_dim
        self.input_size = input_size
        self.activation = activation
        
    def forward(self, input_matrix):

        x = self.h1(input_matrix)
        x = self.h2(x)
        
        if self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'softmax':
            return torch.softmax(x, 0)
        else: 
            return x