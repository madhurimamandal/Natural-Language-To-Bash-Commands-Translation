import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device

        self.layer = nn.TransformerEncoderLayer(hid_dim, 
                                                n_heads, 
                                                pf_dim,
                                                dropout)
        self.encoder = nn.TransformerEncoder(self.layer, n_layers)
        
    def forward(self, src, src_mask):
        
        src = self.encoder(src=src, src_key_padding_mask = src_mask)
            
        return src


class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        self.layers = nn.TransformerDecoderLayer(hid_dim,
                                    n_heads, 
                                    pf_dim, 
                                    dropout)

        self.decoder = nn.TransformerDecoder(self.layers, n_layers)
   
        
    def forward(self, trg, enc_src, trg_pad_mask, trg_sub_mask, src_mask):
                
        output = self.decoder(tgt = trg,
                              memory = enc_src,
                              tgt_mask = trg_sub_mask,                             
                              tgt_key_padding_mask = trg_pad_mask,
                              memory_key_padding_mask = src_mask)
        
            
        return output



class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):

        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
            
        
        
        return trg_pad_mask, trg_sub_mask

    def forward(self, src, trg):
                
        src_mask = self.make_src_mask(src)
        trg_pad_mask, trg_sub_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        
        output = self.decoder(trg, enc_src, trg_pad_mask, trg_sub_mask, src_mask)
        
        return output