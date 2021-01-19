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



def tokenize_nl(text):
  return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_bash(text):
  return data_tools.bash_tokenizer(text)


def tokenize_w(text):
  return text.split(' ')


def create_dataset():    
    SRC = Field(
        tokenize = tokenize_nl,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True
    )

    TRG = Field(
        tokenize = tokenize_bash,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True
    )

    UT = Field(tokenize=tokenize_w, lower=True, pad_token='none')

    

    train = data.TabularDataset(
    path=config.TRAIN_PATH, format=config.X_FORMAT,
    fields=[('source', SRC),
            ('targets', TRG),
            ('utilities', UT)])


    test = data.TabularDataset(
    path=config.TEST_PATH, format=config.X_FORMAT,
    fields=[('source', SRC),
            ('targets', TRG),
            ('utilities', UT)])


    
    valid = data.TabularDataset(
    path=config.VALID_PATH, format=config.X_FORMAT,
    fields=[('source', SRC),
            ('targets', TRG),
            ('utilities', UT)])


    SRC.build_vocab(train, 
                    vectors=torchtext.vocab.Vectors(config.GLOVE_PATH))

    TRG.build_vocab(train)

    UT.build_vocab(train)

    return train, valid, test, SRC, TRG, UT