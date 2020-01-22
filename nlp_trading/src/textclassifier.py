
# -*- coding: utf-8 -*-

import pickle
import nltk
import os
import random
import re
import torch
import logging
import torch.nn.functional as F
from util import *

logger = logging.getLogger('nlp')
       
class TextClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
        """
        Initialize the model by setting up the layers.
        
        Parameters
        ----------
            vocab_size : The vocabulary size.
            embed_size : The embedding layer size.
            lstm_size : The LSTM layer size.
            output_size : The output size.
            lstm_layers : The number of LSTM layers.
            dropout : The dropout probability.
        """
        
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        

        # Setup embedding layer
        self.embedding = torch.nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        
        # Setup additional layers
        self.lstm = torch.nn.LSTM(input_size=self.embed_size, 
                            hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, 
                            batch_first=False, 
                            dropout=self.dropout)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.fc = torch.nn.Linear(in_features=self.lstm_size, out_features=self.output_size)
        self.log_smax = torch.nn.LogSoftmax(dim=1)
        


    def init_hidden(self, batch_size):
        """ 
        Initializes hidden state
        
        Parameters
        ----------
            batch_size : The size of batches.
        
        Returns
        -------
            hidden_state
            
        """
        
        
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if False:#torch.cuda.is_available():
            hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().cuda(),
                  weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().cuda())
        else:
            hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                      weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())
        
        return hidden


    def forward(self, nn_input, hidden_state):
        """
        Perform a forward pass of our model on nn_input.
        
        Parameters
        ----------
            nn_input : The batch of input to the NN.
            hidden_state : The LSTM hidden state.

        Returns
        -------
            logps: log softmax output
            hidden_state: The new hidden state.

        """
        embed = self.embedding(nn_input)
        lstm_out, hidden_state = self.lstm(embed, hidden_state)
        lstm_out = lstm_out[-1]
        #lstm_out = lstm_out.contiguous().view(-1, self.lstm_size)
        logps = self.log_smax(self.dropout(self.fc(lstm_out)))
        #logps = logps[-1]
        
        
        return logps, hidden_state