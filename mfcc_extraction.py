import numpy as np
import torch
from torch.utils import data
# for accesing files in the directory
import glob
import errno


train_audio_path = "./timit_data/data/TRAIN/*/*/*.WAV.wav"  #"./timit_data/data/TRAIN/*/*/*.TXT"
train_audio_filenames = glob.glob(train_audio_path)
train_text_filenames = [filename.replace("WAV.wav", "TXT") for filename in train_audio_filenames]

