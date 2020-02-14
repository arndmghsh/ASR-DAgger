import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import optim


import random
from nltk.metrics.distance import edit_distance

# for Dataloader
from torch.utils import data
import my_data

# for accesing files in the directory
import glob
import errno

from models import Seq2Seq_ScheduledSampling


def train(model, train_dataloader, test_dataloader):

    optimizer = torch.optim.Adam(model.parameters())
    # Default parameters: lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    train_cerloss = []
    test_cerloss = []
    
    beta = 1   # All oracle
#     beta = 0   # All Model
#     beta = 0.75
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
       
        # beta = beta - 0.05
        # beta = np.exp(-epoch)

        # For each iteration, train_dataloader returns the following tuple
        # (input_seqs_padded, input_lengths, target_seqs_padded, target_lengths)
        total_num_batches = 0
        for i_batch, batch in enumerate(train_dataloader):
            input_seqs_padded, target_seqs_padded, target_masks = batch

            optimizer.zero_grad()
            loss = model.forward(input_seqs_padded, target_seqs_padded, target_masks, beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_num_batches+=1
            # print(total_num_batches)
            
        # # Evaluate on the test set and calculate Testing Loss
        # test_loss = 0
        # for j, batch in enumerate(cer_testX):
        #     encoder_hidden = encoder(cer_testX[j])
        #     t_loss = decoder(cer_Y[j], encoder_hidden, cer_M[j], beta)
        #     test_loss+=t_loss
        
        print("Epoch {}: Training Loss: {}".format(epoch, epoch_loss/total_num_batches))

        # print("Epoch {}: Training Loss: {}, Testing Loss: {}".format(epoch, epoch_loss/num_batches, \
        #                                                              test_loss/num_test_batches))
        # train_cerloss.append(epoch_loss/num_batches)
        # test_cerloss.append(test_loss/num_test_batches)
    return train_cerloss, test_cerloss


def evaluate(model, testX, testY):
    model.eval()
    cer = 0
    with torch.no_grad():
        for i, batch in enumerate(testX):
            encoder_output = model.encoder.forward(testX[i])
            prediction = model.decoder.forward_inference(encoder_output)
            # sentence = []
            # for j in prediction:
            #     sentence.append(int2char[int(j)])
            sentence = [int2char[int(j)] for j in prediction]
            sentence = ''.join(sentence)
            cer += edit_distance(sentence, true_label[i]) / len(pred[i])
    avg_cer = cer/len(testX)

    return avg_cer


if __name__ == "__main__":

    EPOCHS = 10
    EMB_SIZE = 256
    HIDDEN_SIZE = 128
    VOCAB_SIZE = 30    #26 + space,sos,eos,pad
    
    # Label/Character encoding
    chars = ['<PAD>','<SOS>', '<EOS>',' ',"a","b","c","d","e","f","g","h","i","j","k", \
             "l","m","n","o","p","q","r","s","t","u", "v","w","x","y","z"]
    int2char = dict(enumerate(chars))
    char2int = {ch:i for i,ch in int2char.items()}

    # Build a custom dataset
    train_dataset = my_data.Dataset('./asr_data/train/', char2int)
    test_dataset = my_data.Dataset('./asr_data/test/', char2int)

    # data loader for custom dataset
    # this will return (input_seqs_padded, input_lengths, target_seqs_padded, target_lengths) for each iteration
    # please see collate_fn for details
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=32,
                                              shuffle=True,
                                              collate_fn=my_data.collate_fn)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=32,
                                              shuffle=True,
                                              collate_fn=my_data.collate_fn)
    
    model = Seq2Seq_ScheduledSampling(VOCAB_SIZE)

    cer_tr, cer_ts = train(model, train_dataloader, test_dataloader)

    print(evaluate(encoder, decoder, testX, 0))

