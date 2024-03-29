import torch

device = torch.device('cuda')
BATCH_SIZE = 8
LEARNING_RATE=0.0001
N_EPOCHS = 10
CLIP = 1
TRAIN_PATH = '../data/train.csv'
VALID_PATH = '../data/valid.csv'
TEST_PATH = '../data/test.csv'
X_FORMAT='csv'
GLOVE_PATH = '../glove.840B.300d.txt'
TRAIN_SPLIT=0.7
TEST_SPLIT=0.15
VALID_SPLIT=0.15
ENCODER_EMBEDDING_DIMENSION=300
DECODER_EMBEDDING_DIMENSION=300
LSTM_HIDDEN_DIMENSION = 512
LSTM_LAYERS=2
ENCODER_DROPOUT=0.5
DECODER_DROPOUT=0.5
LEARNING_RATE=0.001
TEST_MODEL=MODEL_SAVE_FILE='model_seq2seq_1.bin'
RESULTS_SAVE_FILE='results_1.txt'
TEST_RESULTS_FILE='test_1.txt'

