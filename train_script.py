import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data

import numpy as np
import random
from nltk.metrics.distance import edit_distance
# for accesing files in the directory
import glob
import errno

from models import Seq2Seq_ScheduledSampling
import my_data

def train(model, train_dataloader, test_dataloader, test_evaluate_dataloader):
    optimizer = torch.optim.Adam(model.parameters())
    # Default parameters: lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    train_loss_per_epoch = []
    test_loss_per_epoch = []
    test_CER = []
    
    beta = 1  # = All oracle,       beta = 0  = All Model,       beta = 0.75
    for epoch in range(EPOCHS):
        model.train()
        # beta = beta - 0.05
        # beta = np.exp(-epoch)

        # For each iteration, train_dataloader returns the following tuple
        # (input_seqs_padded, input_lengths, target_seqs_padded, target_lengths)
        train_loss = 0.0
        total_num_batches = 0
        for i_batch, batch in enumerate(train_dataloader):
            input_seqs_padded, target_seqs_padded, target_masks = batch

            optimizer.zero_grad()
            loss = model.forward(input_seqs_padded, target_seqs_padded, target_masks, beta)
            # The loss returned is averaged loss per batch
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_num_batches+=1
            # print("Training batch: ", total_num_batches)
        avg_train_loss = train_loss/total_num_batches
        train_loss_per_epoch.append(avg_train_loss)

        # Calculate loss on the Test set
        model.eval()
        test_loss = 0.0
        total_num_test_batches = 0
        for i_batch, batch in enumerate(test_dataloader):
            input_seqs_padded, target_seqs_padded, target_masks = batch

            t_loss = model.forward(input_seqs_padded, target_seqs_padded, target_masks, beta)
            test_loss += t_loss.item()
            total_num_test_batches+=1
            # print("Testing batch: ", total_num_test_batches)
        avg_test_loss = test_loss/total_num_test_batches
        test_loss_per_epoch.append(avg_test_loss)

        # Evaluate Character Error rate performance on the Test set
        model.eval()
        avg_char_err_rate = evaluate(model, test_evaluate_dataloader)
        test_CER.append(avg_char_err_rate)

        print("Epoch: {}, Train Loss: {}, Test Loss: {}, Test CER: {}".format( \
                                        epoch, avg_train_loss, avg_test_loss, avg_char_err_rate))
    return train_loss_per_epoch, test_loss_per_epoch, test_CER


def evaluate(model, test_evaluate_dataloader):
    cer = 0
    num_examples = 0
    with torch.no_grad():
        for i_batch, batch in enumerate(test_evaluate_dataloader):
            input_seq, target_seq, target_masks = batch
            prediction_int = model.forward_inference(input_seq)
            sentence = [int2char[int(j)] for j in prediction_int]
            sentence = ''.join(sentence)

            # Shape of target sequence = 1 x L
            target_sentence = [int2char[int(j)] for j in target_seq[0]]
            target_sentence = ''.join(target_sentence)

            cer += edit_distance(sentence, target_sentence)/len(sentence)
            num_examples += 1
    avg_cer = cer/num_examples
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
    # Batch size = 1 during evaluation/inference
    test_evaluate_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              collate_fn=my_data.collate_fn)
    
    model = Seq2Seq_ScheduledSampling(VOCAB_SIZE)
    loss_tr, loss_ts, test_cher = train(model, train_dataloader, test_dataloader, test_evaluate_dataloader)
