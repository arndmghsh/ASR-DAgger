import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import optim
import random

# for accesing files in the directory
import glob
import errno

# Create a class Encoder, which inherits the properties and methods from the parent class nn.Module
class EncoderLSTM(nn.Module):
    def __init__(self):
        super(EncoderLSTM, self).__init__()
        self.input_size = 20   # given 20 x L acoustic inputs
        self.hidden_size = 128
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first = True)
        
    def forward(self, inp):
        # input to LSTM = B x L x 20
        output, hidden = self.lstm(inp)
        # output, (h_n, c_n) = self.lstm(embedding, (h, c)) ----- (h,c) initialized to zero
        # output size = B x Lx 128
        # (h,c) are from the last time step: both have size [1,B,128]
        # return the last hidden output 1 x B x H
        return (hidden[0][0,:,:],hidden[1][0,:,:])
        

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size):
        super(DecoderLSTM, self).__init__()
        self.embedding_size = 256
        self.decoder_hidden_size = 128
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)        
        # NOTE: Use LSTM Cell here instead if you want to control the hidden state at each time step.
        self.lstm = nn.LSTMCell(self.embedding_size, self.decoder_hidden_size)
        self.lin = nn.Linear(self.decoder_hidden_size, vocab_size)
       
    def forward_step(self, word_embedding, hidden):
        output, new_cell_state = self.lstm(word_embedding, hidden)
        new_hidden = output
        vocab_distrbtn = F.softmax(self.lin(output), dim=1)
        return vocab_distrbtn, (new_hidden, new_cell_state)
        
    def forward(self, inpt, encoder_hidden, mask_Y, beta):
        t_max = inpt.shape[1]
        batch_size = inpt.shape[0]

        loss = 0
        # SOS = 1
        word = inpt[:,0]
        word_embedding = self.embedding(word)

        hidden = encoder_hidden
        for t in range(t_max-1):
            vocab_dist, hidden = self.forward_step(word_embedding, hidden)  # vocab_dist = B x V = 10 x 30
            word = torch.argmax(vocab_dist, dim=1)   # word = B x 1
            
            # DAgger policy = beta*oracle + (1-beta)*model
            u = random.uniform(0, 1)
            if u<=beta:
                # Teacher Forcing
                word_embedding = self.embedding(inpt[:,t+1])
            else:
                # Model's output as next input
                word_embedding = self.embedding(word)
            
            # Cross Entropy Loss
            # ground truth B x 1 is the char at time step t+1 or t+1th column in B x L = 32 x L
            true_label = inpt[:,t+1]            
            # one hot encode the true label # B x 1 = 32 x 1 --> 32 x 30
            onehot = torch.zeros((batch_size, 30))
            for i in range(batch_size):
                onehot[i][true_label[i]]=1
            # Cross entropy loss: vocab_dist 32 x 30, onehot 32 x 30
            NLL = (-1)*torch.log(vocab_dist)
            ce_loss = torch.sum(NLL*onehot, dim=1)
            loss += torch.sum(ce_loss*mask_Y[:,t])
            
        # averaged loss over the entire batch (except padding)
        return loss/torch.sum(mask_Y)

    def forward_inference(self, encoder_output):
    	t_max = 80
    	prediction = []
    	# SOS = 1
    	word = torch.ones(1)
    	word = word.type(torch.LongTensor)
    	word_embedding = self.embedding(word)
    	# Feed in the encoder_hidden
    	hidden = encoder_output
    	for t in range(t_max-1):
    		vocab_dist, hidden = self.forward_step(word_embedding, hidden)  # vocab_dist = B x V = 10 x 30
    		word = torch.argmax(vocab_dist, dim=1)   # word = B x 1
    		if word==2:
    			break
    		prediction.append(word)
    		# Model's output as next input
    		word_embedding = self.embedding(word)

    	return prediction


class Seq2Seq_ScheduledSampling(nn.Module):
	def __init__(self, vocab_size):
		super(Seq2Seq_ScheduledSampling, self).__init__()
		self.encoder = EncoderLSTM()
		self.decoder = DecoderLSTM(vocab_size)

	def forward(self, X_acoustic, Y_labels, Y_mask, beta):
		encoder_output = self.encoder.forward(X_acoustic)
		loss = self.decoder.forward(Y_labels, encoder_output, Y_mask, beta)

		return loss



