import config
import dataset
import model
import engine
import time
from tqdm import tqdm
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
    train, valid, test, SRC, TRG = dataset.create_dataset()
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train, valid, test),
        sort_key=lambda x: len(x.source),
        batch_size=config.BATCH_SIZE,
        device=config.device
        )
    
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = model.Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              config.device)

    dec = model.Decoder(OUTPUT_DIM, 
                HID_DIM, 
                DEC_LAYERS, 
                DEC_HEADS, 
                DEC_PF_DIM, 
                DEC_DROPOUT, 
                config.device)


    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]


    model_tr = model.Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, config.device).to(config.device)

    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    model_tr.apply(initialize_weights)

    optimizer = optim.Adam(model_tr.parameters(), lr=config.LEARNING_RATE)

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    if(args.action=='train'):
        best_valid_loss = float('inf')

        for epoch in tqdm(range(config.N_EPOCHS)):
            
            start_time = time.time()
            
            train_loss = config.train(model_tr, train_iterator, optimizer, criterion, config.CLIP)
            valid_loss = config.evaluate(model_tr, valid_iterator, criterion)
            
            end_time = time.time()
            
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model_tr.state_dict(), 'model.bin')
        
            with open(config.RESULTS_SAVE_FILE, 'a') as f:
                print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s', file=f)
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}', file=f)
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}', file=f)
    
    elif(args.action=='test'):  
        model_tr.load_state_dict(torch.load('model.bin'))

        test_loss, t, o = engine.test(model_tr, test_iterator, criterion, TRG)

        metric_val=0

        for i in range(len(t)):
            metric_val = metric_val + metric_utils.compute_metric(o[i], 1.0, t[i])

        print('Nl2Cmd Metric  | ', metric_val/len(t))

        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

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

