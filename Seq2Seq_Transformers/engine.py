import torch
import torch.nn as nn
import pickle
import config
from tqdm import tqdm


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg[:,:-1])
            
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)



def test(model, iterator, criterion, TRG):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        target_sentences = []
        output_sentences = []
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            b = src.shape[0]
            l = src.shape[1]



            output= model(src, trg[:,:-1])

            out = torch.max(torch.tensor(output), 2)[1]

            tr = trg.cpu().detach().numpy().tolist()
            out = out.cpu().detach().numpy().tolist()

            for k in range(b):
              temp_t = []
              temp_o = []
              for l in range(l-1):
                if(TRG.vocab.itos[tr[k][l]]!='<eos>' and TRG.vocab.itos[tr[k][l]]!='<sos>' and TRG.vocab.itos[tr[k][l]]!='<pad>'):
                  temp_t.append(TRG.vocab.itos[tr[k][l]])
                if(TRG.vocab.itos[out[k][l]]!='<eos>' and TRG.vocab.itos[out[k][l]]!='<sos>' and TRG.vocab.itos[out[k][l]]!='<pad>'):
                  temp_o.append(TRG.vocab.itos[out[k][l]])
              target_sentences.append(' '.join(m for m in temp_t))
              output_sentences.append(' '.join(m for m in temp_o))
               
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator), target_sentences, output_sentences