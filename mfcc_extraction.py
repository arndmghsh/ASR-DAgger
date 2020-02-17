import numpy as np
import torch
from torch.utils import data
# For accessing files in the directory
import glob
import errno

import os, sys
import audio
import re

# Training data
train_audio_path = "./timit_data/data/TRAIN/*/*/*.WAV.wav"  #"./timit_data/data/TRAIN/*/*/*.TXT"
train_audio_filenames = glob.glob(train_audio_path)
train_text_filenames = [filename.replace("WAV.wav", "TXT") for filename in train_audio_filenames]

# Test data
test_audio_path = "./timit_data/data/TEST/*/*/*.WAV.wav"
test_audio_filenames = glob.glob(test_audio_path)
test_text_filenames = [filename.replace("WAV.wav", "TXT") for filename in test_audio_filenames]


mfcc_dir_train = "./timit_data/mfcc_data/train"
for wav_fname in train_audio_filenames:
	fullname = wav_fname.split('/')[-1]
	print(fullname)
	fname = fullname.split('.')[0]
	print(fname)
	break
	# wav_fname = wav_dir + '/' + fname + '.wav'
	wav = audio.load_wav(wav_fname)
	mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
	mspec_fname = mfcc_dir_train + '/' + fname + '.feats'
	np.save(mspec_fname, mel_spectrogram.T, allow_pickle=False)