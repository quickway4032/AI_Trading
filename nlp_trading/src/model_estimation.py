#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from textclassifier import TextClassifier
from collections import Counter
import pickle
from tqdm import tqdm
import nltk
import os
import random
import re
import torch
import torch.nn.functional as F
import logging
from util import *
logger = logging.getLogger('nlp')
nltk.download('wordnet')

class Model:
    
    def __init__(self, data_directory, output_directory):
    
        self.data_directory = data_directory
        self.output_directory = output_directory
        
        self.epochs = 2
        self.batch_size = 512
        self.sequence_length = 40
        self.learning_rate = 1e-3
        self.clip = 5
        self.best_val_acc = 0
        self.print_every = 100
                    
    def read_data(self):
    
        with open('data/twits.pickle', 'rb') as f:
            twits = pickle.load(f)
        
        return twits['data']
    
    def message_sentiment_split(self):

        twits = self.read_data()
        messages = [twit['message_body'] for twit in twits]
        sentiments = [twit['sentiment'] + 2 for twit in twits]
        
        return messages, sentiments

    def preprocess(self, message):
    
    
        # lower case
        text = message.lower()
        
        # Replace URLs with a space in the message
        text = re.sub(r'https?://[^\s]+', ' ', text)
        
        # Replace ticker symbols with a space. The ticker symbols are any stock symbol that starts with $.
        text = re.sub(r'\$[a-zA-Z0-9]*', ' ', text)
        
        # Replace StockTwits usernames with a space. The usernames are any word that starts with @.
        text = re.sub(r'@[a-zA-Z0-9]*', ' ', text)

        # Replace everything not a letter with a space
        text = re.sub(r'[^a-z]', ' ', text)

        # Tokenize by splitting the string on whitespace into a list of words
        tokens = text.split()

        # Lemmatize words using the WordNetLemmatizer. You can ignore any word that is not longer than one character.
        wnl = nltk.stem.WordNetLemmatizer()
        tokens = [wnl.lemmatize(w) for w in tokens if len(w)  > 1]
    
        return tokens

    def bag_of_words(self):

        messages, sentiments = self.message_sentiment_split()
        tokenized = [self.preprocess(message) for message in messages]

        # Remove empty tokenized messages and align the labels
        good_tokens = [idx for idx, token in enumerate(tokenized) if len(token) > 0]
        tokenized = [tokenized[idx] for idx in good_tokens]
        sentiments = [sentiments[idx] for idx in good_tokens]

        self.stacked_tokens = [word for twit in tokenized for word in twit]
        self.bow = Counter(self.stacked_tokens)
        self.tokenized = tokenized
        self.sentiments = sentiments


    def filtered_words(self, low_cutoff, high_cutoff):

        total_num_words = len(self.stacked_tokens)

        freqs = {key: value/total_num_words for key, value in self.bow.items()}

        K_most_common = [word[0] for word in self.bow.most_common(high_cutoff)]

        filtered_words = [word for word in freqs if (freqs[word] > low_cutoff and word not in K_most_common)]
        
        return filtered_words


    def filtered(self):

        filtered_words = self.filtered_words(5e-6, 15)
        # A dictionary for the `filtered_words`. The key is the word and value is an id that represents the word. 
        vocab = {word: ii+1 for ii, word in enumerate(filtered_words)}
        # Reverse of the `vocab` dictionary. The key is word id and value is the word. 
        id2vocab = {ii: word for ii, word in enumerate(filtered_words)}
        # tokenized with the words not in `filtered_words` removed.
        filtered = []
        for twit in tqdm(self.tokenized):
            filtered.append([word for word in twit if word in filtered_words])

        self.vocab = vocab
        self.id2vocab = id2vocab
        self.filtered = filtered

    def balance_class(self):

        balanced = {'messages': [], 'sentiments':[]}

        n_neutral = sum(1 for each in self.sentiments if each == 2)
        N_examples = len(self.sentiments)
        keep_prob = (N_examples - n_neutral)/4/n_neutral

        for idx, sentiment in enumerate(self.sentiments):
            message = self.filtered[idx]
            if len(message) == 0:
                # skip this message because it has length zero
                continue
            elif sentiment != 2 or random.random() < keep_prob:
                balanced['messages'].append(message)
                balanced['sentiments'].append(sentiment) 

        self.balanced = balanced
        self.token_ids = [[self.vocab[word] for word in message] for message in balanced['messages']]
        self.sentiments = balanced['sentiments']
        
    def dataloader(self, messages, labels, sequence_length=30, batch_size=32, shuffle=False):
        """ 
        Build a dataloader.
        """
        if shuffle:
            indices = list(range(len(messages)))
            random.shuffle(indices)
            messages = [messages[idx] for idx in indices]
            labels = [labels[idx] for idx in indices]
    
        total_sequences = len(messages)
    
        for ii in range(0, total_sequences, batch_size):
            batch_messages = messages[ii: ii+batch_size]
            
            # First initialize a tensor of all zeros
            batch = torch.zeros((sequence_length, len(batch_messages)), dtype=torch.int64)
            for batch_num, tokens in enumerate(batch_messages):
                token_tensor = torch.tensor(tokens)
                # Left pad!
                start_idx = max(sequence_length - len(token_tensor), 0)
                batch[start_idx:, batch_num] = token_tensor[:sequence_length]
            
            label_tensor = torch.tensor(labels[ii: ii+len(batch_messages)])
            
            yield batch, label_tensor
        
    def train_valid_split(self):
        
        valid_split = int(0.8*len(self.token_ids))
        self.train_features = self.token_ids[:valid_split]
        self.valid_features = self.token_ids[valid_split:]
        self.train_labels = self.sentiments[:valid_split]
        self.valid_labels = self.sentiments[valid_split:]
        
    def model_build(self): 
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        configs = get_configs()
        hp_dict = configs['hyperparams']
        
        self.vocab_size = len(self.vocab)+1
        self.embed_size = hp_dict['embed_size']
        self.lstm_size = hp_dict['lstm_size']
        self.output_size: hp_dict['output_size']
        self.lstm_layers: hp_dict['lstm_layers']
        self.dropout = hp_dict['dropout']
        
        model = TextClassifier(self.vocab_size, self.embed_size, self.lstm_size, self.output_size, self.lstm_layers, self.dropout)
        model.embedding.weight.data.uniform_(-1, 1)
        model.to(self.device)
        
        return model

    def model_train(self):
    
        model = self.model_build()
        
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        model.train()

        for epoch in range(self.epochs):
            logger.info(f'Starting epoch {epoch+1}')
            steps = 0        
            hidden = model.init_hidden(self.batch_size)
            for text_batch, labels in self.dataloader(
                    self.train_features, self.train_labels, batch_size=self.batch_size, sequence_length=self.sequence_length, shuffle=True):
                if text_batch.size() != torch.Size([self.sequence_length, self.batch_size]):
                    continue
                steps += 1
                hidden = tuple([each.data for each in hidden])
                
                # Set Device
                text_batch, labels = text_batch.to(self.device), labels.to(self.device)
                for each in hidden:
                    each.to(self.device)
                
                model.zero_grad()
                log_ps, hidden = model.forward(text_batch, hidden)
                loss = criterion(log_ps, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                optimizer.step()
                
                if steps % self.print_every == 0:
                    model.eval()
                    val_losses = []
                    val_accuracy = []
                    val_hidden = model.init_hidden(self.batch_size)
        
                    for val_text_batch, val_labels in self.dataloader(
                    self.valid_features, self.valid_labels, batch_size=self.batch_size, sequence_length=self.sequence_length):
                        if val_text_batch.size() != torch.Size([self.sequence_length, self.batch_size]):
                            continue
                        val_text_batch, val_labels = val_text_batch.to(self.device), val_labels.to(self.device)
                        val_hidden = tuple([each.data for each in val_hidden])
                        for each in val_hidden:
                            each.to(self.device)
                        val_log_ps, hidden = model.forward(val_text_batch, val_hidden)
                        val_loss = criterion(val_log_ps.squeeze(), val_labels)
                        val_losses.append(val_loss.item())
                        
                        val_ps = torch.exp(val_log_ps)
                        top_p, top_class = val_ps.topk(1, dim=1)
                        equals = top_class == val_labels.view(*top_class.shape)
                        val_accuracy.append(torch.mean(equals.type(torch.FloatTensor)).item())
        
                    model.train()
                    this_val_acc = sum(val_accuracy)/len(val_accuracy)
                    
                    logger.info(f'Epoch: {epoch+1}/{self.epochs}...',
                          f'Step: {steps}...',
                          f'Loss: {loss.item()}...',
                          f'Val Loss: {sum(val_losses)/len(val_losses)}',
                          f'Val Accuracy: {this_val_acc}')
                    if this_val_acc > self.best_val_acc:
                        torch.save({
                    'epoch': epoch,
                    'step': steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'vocab': self.vocab,
                    'inputs': [self.vocab_size, self.embed_size, self.lstm_size, self.output_size, self.lstm_layers, self.dropout]
                    }, './model/best_model')
                        self.best_val_acc = this_val_acc
                        logger.info('New best accuracy - model saved')
                        

                        
def run(data_directory, output_directory):

    md = Model(data_directory, output_directory)
    logger.info('Loading Data')
    md.read_data()
    logger.info('Bag of words')
    md.bag_of_words()
    md.filtered()
    md.balance_class()
    md.train_valid_split()
    md.model_build()
    md.model_train()    