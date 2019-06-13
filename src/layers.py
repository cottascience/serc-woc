import torch, sys
from torch.nn import Linear
import torch.nn.functional as F
from torch import nn


class DeepSets(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_size):
        super(DeepSets,self).__init__()
        self.f = Linear(input_size+embedding_size, hidden_size)
        self.rho = Linear(hidden_size, output_size)
    def forward(self, x, idxs, embeddings):
        x = F.relu(self.f(torch.cat((x,embeddings),dim=1)))
        x = torch.cumsum(x,dim=0)
        agg_x = x[(idxs[0][1])-1:(idxs[0][1])]
        end = 0
        start = idxs[0][1]
        idxs = idxs[1:]
        for rows in idxs:
            end = start + rows[1]
            agg_x = torch.cat((agg_x, x[(end-1):end] - x[(start-1):start]),dim=0)
            start = end
        return F.relu(self.rho(agg_x))
