import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data

import numpy as np
import tensorboard_logger
import random
from nltk.metrics.distance import edit_distance
# for accesing files in the directory
import glob
import errno
import sys

from models import Seq2Seq_ScheduledSampling
import my_data
import utils

def train(model, train_dataset, test_dataset):
    checkpoint_interval = 5
    checkpoint_dir = './checkpoints/'

    # Data loaders - returns (input_seqs_padded, input_lengths, target_seqs_padded, target_lengths) for each iteration
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=32,
                                              shuffle=True,
                                              collate_fn=my_data.collate_fn)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=32,
                                              shuffle=True,
                                              collate_fn=my_data.collate_fn)

    optimizer = torch.optim.Adam(model.parameters())
    # Default parameters: lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    # train_loss_per_epoch = []
    # test_loss_per_epoch = []
    # test_CER = []
    
    beta = 1  # = All oracle,       beta = 0  = All Model,       beta = 0.75
    for epoch in range(EPOCHS):
        model.train()
        # beta = beta - 0.05
        # beta = np.exp(-epoch)

        # For each iteration, train_dataloader returns the following tuple
        # (input_seqs_padded, input_lengths, target_seqs_padded, target_lengths)
        train_loss = 0.0
        total_num_batches = 0
        # for i_batch, (input_seqs_padded, target_seqs_padded, target_masks) in tqdm(enumerate(train_dataloader)):
        for i_batch, batch in enumerate(train_dataloader):
            input_seqs_padded, target_seqs_padded, target_masks = batch

            optimizer.zero_grad()
            loss = model.forward(input_seqs_padded, target_seqs_padded, target_masks, beta) # averaged loss per batch
            loss.backward()
            train_loss += loss.item()
            # grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), clip_thresh=1.0)
            optimizer.step()
            total_num_batches+=1
        avg_train_loss = train_loss/total_num_batches
        # train_loss_per_epoch.append(avg_train_loss)

        # Save model after few epochs
        if epoch > 0 and epoch % checkpoint_interval == 0: 
            utils.save_checkpoint(model, optimizer, epoch, checkpoint_dir)

        # Calculate loss on the Test set
        model.eval()
        test_loss = 0.0
        total_num_test_batches = 0
        for i_batch, batch in enumerate(test_dataloader):
            input_seqs_padded, target_seqs_padded, target_masks = batch

            t_loss = model.forward(input_seqs_padded, target_seqs_padded, target_masks, beta)
            test_loss += t_loss.item()
            total_num_test_batches+=1
        avg_test_loss = test_loss/total_num_test_batches
        # test_loss_per_epoch.append(avg_test_loss)

        # Evaluate Character Error Rate on the Test set
        model.eval()
        avg_char_err_rate = evaluate(model, test_dataset)
        # test_CER.append(avg_char_err_rate)

        # Log the values
        tensorboard_logger.log_value("Train_loss", float(avg_train_loss), epoch)
        tensorboard_logger.log_value("Test_loss", float(avg_test_loss), epoch)
        tensorboard_logger.log_value("Test_CER", float(avg_char_err_rate), epoch)
        # log_value("gradient norm", grad_norm, epoch)
        # log_value("learning rate", current_lr, global_step)

        print("Epoch: {}, Train Loss: {}, Test Loss: {}, Test CER: {}".format( \
                                        epoch, avg_train_loss, avg_test_loss, avg_char_err_rate))
    return


def evaluate(model, test_dataset):
    # Dataloader, Batch size = 1 during evaluation/inference
    dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              collate_fn=my_data.collate_fn)
    cer = 0
    num_examples = 0
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
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

    # Setup tensorboard logger
    log_path = "./runs/run0"
    tensorboard_logger.configure(log_path)

    EPOCHS = 50
    EMB_SIZE = 256
    HIDDEN_SIZE = 128
    VOCAB_SIZE = 30    #26 + space,sos,eos,pad
    
    # Label/Character encoding
    chars = ['<PAD>','<SOS>', '<EOS>',' ',"a","b","c","d","e","f","g","h","i","j","k", \
             "l","m","n","o","p","q","r","s","t","u", "v","w","x","y","z"]
    int2char = dict(enumerate(chars))
    char2int = {ch:i for i,ch in int2char.items()}

    # Custom dataset
    train_dataset = my_data.Dataset('./asr_data/train/', char2int)
    test_dataset = my_data.Dataset('./asr_data/test/', char2int)
    
    model = Seq2Seq_ScheduledSampling(VOCAB_SIZE)
    train(model, train_dataset, test_dataset)

    sys.exit(0)
