import config
import dataset
import model
import engine
import utils
import time
from torchtext.data import BucketIterator
import torch.optim as optim
import torch.nn as nn
import torch
import math
import random
import numpy as np
import argparse
from torchtext.data.metrics import bleu_score
import  metric_utils
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--action", 
	type=str, 
	default='train', 
	help="whether to train or test")

args = parser.parse_args()
def run():
    Seed = 1234
    random.seed(Seed)
    np.random.seed(Seed)
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.backends.cudnn.deterministic = True
    train, valid, test, SRC, TRG, UT = dataset.create_dataset()
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train, valid, test),
        sort_key=lambda x: len(x.source),
        batch_size=config.BATCH_SIZE,
        device=config.device
        )
    
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    
    ENC_EMB_DIM = config.ENCODER_EMBEDDING_DIMENSION
    DEC_EMB_DIM = config.DECODER_EMBEDDING_DIMENSION
    HID_DIM = config.LSTM_HIDDEN_DIMENSION
    N_LAYERS = config.LSTM_LAYERS
    ENC_DROPOUT = config.ENCODER_DROPOUT
    DEC_DROPOUT = config.DECODER_DROPOUT

    utilities = pickle.load(open('../list_of_utilities.pkl', 'rb'))

    utilities_dict = {}
    
    for i in range(len(utilities)):
        utilities_dict[i] = TRG.vocab.stoi[utilities[i]]


    attn = model.Attention(HID_DIM, HID_DIM)
    enc = model.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, HID_DIM, ENC_DROPOUT, len(utilities), utilities_dict)
    dec = model.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, HID_DIM, DEC_DROPOUT, attn)

    model_rnn = model.Seq2Seq(enc, dec, config.device).to(config.device)

    optimizer = optim.Adam(model_rnn.parameters(), lr=config.LEARNING_RATE)

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    if(args.action=='train'):
        model_rnn.apply(utils.init_weights)

        best_valid_loss = float('inf')

        for epoch in range(config.N_EPOCHS):    
            start_time = time.time()

            train_loss = engine.train_fn(model_rnn, train_iterator, optimizer, criterion, config.CLIP, UT)
            valid_loss = engine.evaluate_fn(model_rnn, valid_iterator, criterion, UT)

            end_time = time.time()

            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model_rnn.state_dict(), config.MODEL_SAVE_FILE)

            with open(config.RESULTS_SAVE_FILE, 'a') as f:
                print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s', file=f)
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}', file=f)
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}', file=f)
    
    elif(args.action=='test'):  
        model_rnn.load_state_dict(torch.load(config.TEST_MODEL))      
        loss, target, output = engine.test_fn(model_rnn, test_iterator, criterion, SRC, TRG)
        bl = bleu_score(
            output, target, max_n=1, weights=[1]
            )
        met = 0
        
        for z in range(len(output)):
            out = ' '.join(output[z][y] for y in range(min(10, len(output[z]))))
            tar = ' '.join(y for y in target[z])

            met = met + metric_utils.compute_metric(out, 1.0, tar) 
            
        with open(config.TEST_RESULTS_FILE, 'w') as f:
            print(f'Test bleu :, {bl*100}, Test PPL: {math.exp(loss):7.3f}', 'Metric:',
                met/len(output), file=f)

    elif(args.action=='save_vocab'):
        print('Source Vocab Length', len(SRC.vocab))
        print('Target vocab length', len(TRG.vocab))
        s1 = '\n'.join(k for k in SRC.vocab.itos)
        s2 = '\n'.join(k for k in TRG.vocab.itos)
        with open('NL_vocabulary.txt', 'w') as f:
            f.write(s1)
        with open('Bash_vocabulary.txt', 'w') as f:
            f.write(s2)
        
        


if __name__=='__main__':
    run()

