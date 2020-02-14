import numpy as np
import torch
from torch.utils import data
# for accesing files in the directory
import glob
import errno


class Dataset(data.Dataset):
    def __init__(self, path, char2int):
        
        self.char2int = char2int

        input_path = path+"*.npy"  #'./asr_data/train/*.npy'
        input_filenames = glob.glob(input_path)
        self.input_seqs = [torch.from_numpy(np.transpose(np.load(filename))) for filename in input_filenames]
        # Shape of .npy files is 20 x L, we want B x L x 20

        target_path = path+"*.txt"  #'./asr_data/train/*.txt'
        target_filenames = glob.glob(target_path)
        self.target_seqs = [self.preprocess_target_seq(filename) for filename in target_filenames]

    def __getitem__(self, index):
        # Select a sample pair
        input_seq = self.input_seqs[index]
        target_seq = self.target_seqs[index]

        return input_seq, target_seq

    def __len__(self):
        return len(self.input_seqs)

    def preprocess_target_seq(self, filename):
        # with open(filename) as file:
        #     sentence = file.read()
        file = open(filename)
        sentence = file.read()
        file.close()

        char_seq = ['<SOS>'] + list(sentence) + ['<EOS>']
        encoded_seq = [self.char2int[char] for char in char_seq]
        return torch.LongTensor(encoded_seq)


def collate_fn(batch):
    """
    Args:
        batch: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.
    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length, dim).
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        target_masks: torch tensor of shape (batch_size, padded_length).
    """

    # Sort a list by sequence length (descending order) to use pack_padded_sequence
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    # Seperate source and target sequences
    input_seqs, target_seqs = zip(*batch)
    input_lengths = [len(seq) for seq in input_seqs]
    target_lengths = [len(seq) for seq in target_seqs]

    input_seqs_padded = torch.nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_seqs_padded = torch.nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=0)

    batch_size = target_seqs_padded.shape[0]
    padded_length =  target_seqs_padded.shape[1]
    # Create the mask for the target sequence 
    target_masks = torch.zeros(batch_size, padded_length)
    for i, lengths in enumerate(target_lengths):
        target_masks[i, 0:lengths] = torch.ones(1,lengths)

    return input_seqs_padded, target_seqs_padded, target_masks
