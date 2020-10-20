import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self,embed_dim, hidden_dim, output, drop_out , num_layers):
        
        super().__init__()
  
        self.rnn = nn.LSTM(embed_dim , hidden_dim,dropout=drop_out, num_layers = num_layers, batch_first=True)
    
        self.nf = nn.Linear(hidden_dim, output) 
        
        
    def forward(self, batch):
        
        s , (h, c) = self.rnn(batch)
        outputs = s 
        outputs = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)
        s = self.nf(outputs) 
        return  s

       
    
    
    
    
    
    
    
