#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from textclassifier import TextClassifier
from model_estimation import Model

class Predict:
    
    def __init__(self, text, filepath = './model/best_model'):
        self.text = text
        self.filepath = filepath
        
    def load_checkpoint(self):
        
        self.checkpoint = torch.load(filepath)
        
        vocab_size, embed_size, lstm_size, output_size, lstm_layers, dropout = self.checkpoint['inputs']
        
        model = TextClassifier(vocab_size, embed_size, lstm_size, output_size, lstm_layers, dropout)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model

    def predict(self):
        """ 
        Make a prediction on a single sentence.
        
        Parameters
        ----------
            text : The string to make a prediction on.
            model : The model to use for making the prediction.
            vocab : Dictionary for word to word ids. The key is the word and the value is the word id.
        
        Returns
        -------
            pred : Prediction vector
        """    
        
        model = self.load_checkpoint()
        filtered_words = self.checkpoint['filtered_words']
        vocab = self.checkpoint['vocab']    
        tokens = MODEL.preprocess(self.text)
        
        # Filter non-vocab words
        tokens = [word for word in tokens if word in filtered_words]
        # Convert words to ids
        tokens = [vocab[word] for word in tokens] 
            
        # Adding a batch dimension
        text_input = torch.tensor(tokens).unsqueeze(1)
        # Get the NN output
        #print(text_input.size())
        hidden  = model.init_hidden(text_input.size(1))
        logps, _ = model.forward(text_input, hidden)
        # Take the exponent of the NN output to get a range of 0 to 1 for each label.
        pred = torch.exp(logps)
        
        return pred.detach().numpy()
    

def run(text):

    pt = Predict(text)
    pred = pt.predict()
    print(pred)