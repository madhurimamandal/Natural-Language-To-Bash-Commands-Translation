import config
import torch
import torch.nn as nn
import random
import torch.nn.functional as F


#utd = {utility_num : utility_location_in_vocab}

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, num_ut, utd):
        super().__init__()        
        self.embedding = nn.Embedding(input_dim, emb_dim)        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)        
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(dec_hid_dim, num_ut)
        self.utd = utd
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        ut = self.linear1(hidden)  #[8, 118]

        _, ui = torch.topk(ut, k=2, dim=1)   #[8, 2]

        temp =[]
        for i in range(ui.shape[0]):
            l=[]
            l.append(self.utd[ui[i][0].item()])
            l.append(self.utd[ui[i][1].item()])
            temp.append(l)
        
        uv = self.embedding(torch.LongTensor(temp).to(config.device))       #[8, 2, 300]

        uv = torch.sum(uv, 1)

        return outputs, hidden, uv, ut
        

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        attention = self.v(energy).squeeze(2)
        

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim + emb_dim, output_dim)        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, uv):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)

        uv = uv.unsqueeze(0)

        #embedded = embedded+uv


        rnn_input = torch.cat((embedded, weighted, uv), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        uv = uv.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded, uv), dim = 1))
        
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, uv, ut = self.encoder(src)
                
        input = trg[0,:]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs, uv)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[t] if teacher_force else top1

        return outputs, ut
