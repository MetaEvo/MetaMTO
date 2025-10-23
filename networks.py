import torch
import torch.nn.functional as F
from torch import nn
import math


class Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.queryproj = nn.Linear(embed_dim, embed_dim)
        self.keyproj = nn.Linear(embed_dim, embed_dim)
        self.valueproj = nn.Linear(embed_dim, embed_dim)
        self.norm_factor = 1 / math.sqrt(embed_dim)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        bs, query_length, embed_dim = x.shape
        Q = self.queryproj(x)
        K = self.keyproj(x)
        V = self.keyproj(x)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(1,2)).contiguous()
        mask = torch.eye(query_length, dtype=torch.bool).to(self.device)
        compatibility = torch.masked_fill(compatibility, mask, -torch.inf)
        compatibility = F.softmax(compatibility, dim=-1)

        out = torch.matmul(compatibility, V)

        return compatibility.detach(), out


class Normalization(nn.Module):
    def __init__(self, embed_dim):
        super(Normalization,self).__init__()
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self,x):
        x = x.transpose(1,2).contiguous()
        out = self.norm(x)
        return out.transpose(1,2).contiguous()


class AttentionsubLayer(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionsubLayer, self).__init__()
        self.embed_dim = embed_dim

        self.Att = Attention(embed_dim=embed_dim)
        self.Norm = Normalization(embed_dim)

    def forward(self, input):
        compatibility, out = self.Att(input)
        return compatibility, self.Norm(out + input)


class FFN_sublayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FFN_sublayer, self).__init__()
        self.FF = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm = Normalization(embed_dim)

    def forward(self, x):
        out = self.FF(x)
        out = self.norm(out + x)
        return out


class Encoder(nn.Module):
    def __init__(self,embed_dim,hidden_dim):
        super(Encoder, self).__init__()
        self.Att_sublayer = AttentionsubLayer(embed_dim)
        self.FFN = FFN_sublayer(embed_dim, hidden_dim)

    def forward(self,x):
        compatibility, out = self.Att_sublayer(x)
        out = self.FFN(out)
        return compatibility, out


class EmbeddingNet(nn.Module):

    def __init__(
            self,
            node_dim,
            embedding_dim,
    ):
        super(EmbeddingNet, self).__init__()
        self.embedder = nn.Linear(node_dim, embedding_dim)

    def forward(self, x):
        h_em = self.embedder(x)
        return h_em