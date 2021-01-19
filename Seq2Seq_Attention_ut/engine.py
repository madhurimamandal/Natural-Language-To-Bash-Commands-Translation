import torch
import pickle
import config
import tqdm

def loss_utils(inp, tar):
  return torch.nn.BCEWithLogitsLoss()(inp, tar)

def train_fn(model, iterator, optimizer, criterion, clip, UT):    
  model.train()
  
  epoch_loss = 0
 
  for i, batch in enumerate(iterator):      
      src = batch.source
      trg = batch.targets       #[20, 8]
      uti = batch.utilities     #[3, 8]
      
      optimizer.zero_grad()
      
      output,ut = model(src, trg)   #ut = [8, 118]
   

      utili = pickle.load(open('../list_of_utilities.pkl', 'rb'))

      uti2ind = {utili[k]: k for k in range(len(utili))}

      uti = uti.permute(1, 0)  #uti = [8, 3]

      temp_ut = []

      for k in range(uti.shape[0]):
        temp = [0 for m in range(len(utili))]
        for l in range(uti.shape[1]):
          t = uti[k][l].item()
          u = UT.vocab.itos[t]
          try:
            j = uti2ind[u]
            if(u!='none'):
              temp[j]=temp[j]+1
          except:
            j = uti2ind['none']
            temp[j]=temp[j]+1
            
            
        temp_ut.append(temp)

      temp_ut = torch.Tensor(temp_ut).to(config.device)     #[8, 118]

      output_dim = output.shape[-1]
      
      output = output[1:].view(-1, output_dim)
      trg = trg[1:].view(-1)
      loss = criterion(output, trg) + loss_utils(ut, temp_ut)
      loss1 = criterion(output, trg)
      
      loss.backward()
      
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
      optimizer.step()
      
      epoch_loss += loss1.item()
      
  return epoch_loss / len(iterator)


def evaluate_fn(model, iterator, criterion, UT):    
  model.eval()
  
  epoch_loss = 0
  
  with torch.no_grad():
  
      for i, batch in enumerate(iterator):
          src = batch.source
          trg = batch.targets
          
          output,ut = model(src, trg, 0)

          output_dim = output.shape[-1]
          
          output = output[1:].view(-1, output_dim)
          trg = trg[1:].view(-1)

          loss = criterion(output, trg) 
          
          epoch_loss += loss.item()
      
  return epoch_loss / len(iterator)

def test_fn(model, iterator, criterion, SRC, TRG):    
  model.eval()
  
  epoch_loss = 0
  target_sentences = []
  output_sentences =[]
  
  with torch.no_grad():
  
      for i, batch in enumerate(iterator):
          src = batch.source
          trg = batch.targets
          

          output,ut = model(src, trg, 0)


          actual = trg
          predicted = torch.argmax(output, 2, keepdim=True).squeeze(2)

          actual = actual.permute(1, 0)                       #[8, len]
          predicted = predicted.permute(1, 0)                 #[8, len]


          output_dim = output.shape[-1]          
          output = output[1:].view(-1, output_dim)        
          trg = trg[1:].view(-1)
          loss = criterion(output, trg)

          batch_ = actual.shape[0]
          len_ = actual.shape[1]

          actual = actual.cpu().detach().numpy().tolist()
          predicted = predicted.cpu().detach().numpy().tolist()

          for j in range(batch_):
            ac_temp = []
            for k in range(len_):
              if(TRG.vocab.itos[actual[j][k]]!='<sos>' and TRG.vocab.itos[actual[j][k]]!='<eos>' and TRG.vocab.itos[actual[j][k]]!='<pad>'):
                ac_temp.append(TRG.vocab.itos[actual[j][k]])
            target_sentences.append(ac_temp)

          for j in range(batch_):
            pr_temp = []
            for k in range(len_):
              if(TRG.vocab.itos[predicted[j][k]]!='<sos>' and TRG.vocab.itos[predicted[j][k]]!='<eos>' and TRG.vocab.itos[predicted[j][k]]!='<pad>' and TRG.vocab.itos[predicted[j][k]]!='<unk>'):
                pr_temp.append(TRG.vocab.itos[predicted[j][k]])
            output_sentences.append(pr_temp)
          
          epoch_loss += loss.item()
  
  for z in range(20):
    print('--------------------------------------')
    print('Actual')
    print(' '.join(y for y in target_sentences[z]))
    print('Predicted')
    print(' '.join(y for y in output_sentences[z]))
    print('--------------------------------------')
      
  return epoch_loss / len(iterator), target_sentences, output_sentences

