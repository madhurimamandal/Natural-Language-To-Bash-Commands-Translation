import pandas as pd
import numpy as np
import spacy
import torchtext
from torchtext.data import Field, Dataset
import torchtext.data as data
import sys
sys.path.append('..')
from bashlint import data_tools
import config
spacy_en = spacy.load('en')


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_bash(text):
  return data_tools.bash_tokenizer(text)



def create_dataset():  
    SRC = Field(tokenize = tokenize_en, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True, 
                batch_first = True, fix_length=config.MAX_LEN)

    TRG = Field(tokenize = tokenize_bash, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True, 
                batch_first = True, sequential=True, use_vocab=True, fix_length=config.MAX_LEN)

    

    train = data.TabularDataset(
    path=config.TRAIN_PATH, format=config.X_FORMAT,
    fields=[('src', SRC),
            ('trg', TRG)])

    valid = data.TabularDataset(
        path=config.VALID_PATH, format=config.X_FORMAT,
        fields=[('src', SRC),
                ('trg', TRG)])

    test = data.TabularDataset(
        path=config.TEST_PATH, format=config.X_FORMAT,
        fields=[('src', SRC),
                ('trg', TRG)])


    SRC.build_vocab(train, min_feq=2,
                    vectors=torchtext.vocab.Vectors(config.GLOVE_PATH))

    TRG.build_vocab(train, min_freq=2)

    return train, valid, test, SRC, TRG