import torch
import torch.nn as nn
import torch.nn.functional as F 

#Define a GRU-based Encoder for The Flux Sequence
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, batch_first=True):
        super(Encoder, self).__init__()
        
        #Defines the size of the hidden state
        self.hidden_dim = hidden_dim

        self.n_layers = n_layers

        #Define a GRU module that accepts sequence_length sized inputs
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.3)

        #Define a fully connected layer for generating class score
        self.fc = nn.Linear(hidden_dim, output_dim)

        #Define a ReLU Activation
        self.relu = nn.ReLU()

    #Define forward pass which takes in previous hiddden state
    def forward(self, x, h):
        #Compute output and next hidden state, from previous hidden state, h
        out, h = self.gru(x, h)

        #Compute Output Class Probabilities
        out = self.fc(self.relu(out[:, -1]))
        
        return out, h

    #Initialize a learnable hidden state for the start of an epoch
    def init_hidden(self, batch_size):
        #Extract the parameters of the model 
        weight = next(self.parameters()).data

        #Create a new weight with shape (n_layers, batch_size, hidden_dim)
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to('cpu')
        
        return hidden