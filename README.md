# ASR-DAgger
Seq2Seq Automatic Speech Recognition using DAgger (Scheduled Sampling) and Supervised Imitation Learning (Teacher Forcing).


The effect of Scheduled Sampling on RNN based sequence prediciotn model was first studied in [1] S. Bengio, O. Vinyals, N. Jaitly, and N. Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In Advances in Neural Information Processing Systems, pages 1171–1179, 2015.

It studied the performance of three systems: (1) Image captioning (2) Constituency Parsing (3) Speech Recognition. The speech recognition model was two layer LSTM for direct acoustic to HMM state (phoneme) prediction. The frames to HMM states alighnment and labeling was obtained using KALDI on TIMIT dataset. Note that this was not an encoder-decoder based model.





For logging and visualization on tensorboard:\
tensorboard --logdir runs\
http://localhost:6006

