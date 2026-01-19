import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from torch import nn

class AGE(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()

        self._reset_parameters()
        #
        self.d_model = d_model

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):

        tgt_norm = tgt / tgt.norm(dim=-1, keepdim=True)  # (64, 8, 512)

        cosine_sim_matrix = torch.matmul(tgt_norm, tgt_norm.transpose(1, 2))  # (64, 8, 10)

        inverse_similarity_matrix = cosine_sim_matrix
        inverse_similarity_matrix = torch.tanh(inverse_similarity_matrix)

        weighted_tgt = torch.bmm(inverse_similarity_matrix, tgt)
        output = weighted_tgt

        return output

class AGE_base(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()

        self._reset_parameters()
        #
        self.d_model = d_model
        self.w = nn.Linear(d_model, 1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, tgt):
        """ AGE """
        """
        src:visual
        tgt:targets
        """
        weight=cosine_similarity(tgt,tgt)
        output=torch.bmm(weight, tgt)

        return output

def cosine_similarity(vec1, vec2, flag=None):
    vec1_norm = vec1 / vec1.norm(dim=-1, keepdim=True)  # (64, 8, 512)
    vec2_norm = vec2 / vec2.norm(dim=-1, keepdim=True)  # (64, 8, 512)
    cosine_sim_matrix = torch.bmm(vec1_norm, vec2_norm.transpose(1, 2))  # (64, 8, 10)
    if flag==1:
        for i in range(cosine_sim_matrix.shape[1]):
            cosine_sim_matrix[:, i, i] = 0

    sim_min = cosine_sim_matrix.view(cosine_sim_matrix.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1)
    sim_max = cosine_sim_matrix.view(cosine_sim_matrix.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1)

    normalized_cosine_sim_matrix = (cosine_sim_matrix - sim_min) / (sim_max - sim_min)
    return normalized_cosine_sim_matrix

