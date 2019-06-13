import sys, torch
import torch.nn.functional as F
from torch import nn
from layers import DeepSets
from torch.nn import Linear

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, units, RR=False):
        super(Model,self).__init__()
        self.RR = RR
        self.user_layer = Linear(input_size, hidden_size)
        self.question_layers = nn.ModuleDict()
        for u in units:
            self.question_layers[u] = DeepSets(units[u], hidden_size, 1, hidden_size)
        if RR:
            self.logit = Linear(len(units), 1)
        else:
            self.logit = Linear(len(units)+output_size, output_size)
    def forward(self, x, idxs, units, prev_y):
        usr_embeddings = self.user_layer(x)
        #prev_y = prev_y.view(len(prev_y),1).float()
        prev_y = prev_y.float()
        if self.RR:
            list_of_x = []
        else:
            list_of_x = [prev_y]
        start = 0
        for u in units:
            end = start + units[u]
            list_of_x.append(self.question_layers[u](x[:,start:end], idxs, usr_embeddings))
            start = end
        return self.logit(torch.cat(list_of_x, dim=1))
