import numpy as np
import torch
from torch.utils import data
# For accessing files in the directory
import glob
import errno

import os, sys
import audio
import re

def extract_MFCC_and_text(wav_file_path, mfcc_dir):

	wav_filenames = glob.glob(wav_file_path)

	for wav_fname in wav_filenames:
		text_filename = wav_fname.replace(".WAV.wav", ".TXT")
		fullname = wav_fname.split('/')[-1]
		fname = fullname.split('.')[0]

		# Process the text: remove the first two numbers from the text file
		with open(text_filename, 'r') as file:
			sentence = file.read()
		sentence = sentence.split()[2:] + ['\n']
		sentence = ' '.join(sentence).lower()
		# Write the prcoeesed text to the mfcc directory
		text_fname = mfcc_dir + '/' + fname + '.txt'
		with open(text_fname, "w") as file:
			file.write(sentence)

		# Generate the MFCC features
		wav = audio.load_wav(wav_fname)
		mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
		mspec_fname = mfcc_dir + '/' + fname
		np.save(mspec_fname, mel_spectrogram, allow_pickle=False) #generates features of shape: L x 80

	return


if __name__ == "__main__":

	# Training data
	wav_path_train = "./timit_data/data/TRAIN/*/*/*.WAV.wav"  #"./timit_data/data/TRAIN/*/*/*.TXT"
	mfcc_dir_train = "./timit_data/mfcc_data/train"
	extract_MFCC_and_text(wav_path_train, mfcc_dir_train)

	# Test data
	wav_path_test = "./timit_data/data/TEST/*/*/*.WAV.wav"
	mfcc_dir_test = "./timit_data/mfcc_data/test"	
	extract_MFCC_and_text(wav_path_test, mfcc_dir_test)

