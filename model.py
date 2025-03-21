import torch.nn as nn
import torch.functional as F
from egnn_pytorch import EGNN

class MLP(nn.Module):
   pass

# h2l_edges, h_xs, l_xs, l_edges, edge_xs
# (higher to lower edges), (higher features), (lower featurs), (lower edges), (lower edge features)
class HierarchicalEGNN(nn.Module):
    pass 

class BindNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, pdrop=0.0):
        super().__init__()
        self.n_layers = n_layers

        self.atom_proj = nn.Linear(input_dim, hidden_dim)
        self.sub_emb = nn.Embedding(subgroup_emb_dim, hidden_dim)
        self.aa_emb = nn.Embedding(aa_emb_dim, hidden_dim)

        self.egnn_layers = nn.ModuleList([EGNN(dim=hidden_dim) for _ in range(n_layers)])
        #readout
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.Dropout(pdrop),
                                 nn.SiLU()
                              )
        self.att_W = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hg):
        # Unpacking hierarchical graph in atom, subgroup and aminoacid graphs
        atom_xs, atom_edges, atom_edges_xs, = hg.atom_nodes, hg.atom_edge_index, hg.atom_edge_xs
        sub_xs, sub_edges, g2a_edges, sub_edges_xs = hg.subgroup_nodes, hg.ubgroup_edge_index hg.g2a_index, hg.subgroup_edge_features
        aa_xs, aa_edges, aa2g_edges, aa_edges_xs = hg.aa_nodes, hg.aa_edge_index, hg.aa2g_index, hg.aa_edge_features

        # Project atom features and get embeddings for subgroups and AAs
        atom_xs_proj = self.atom_proj(atom_xs)
        sub_xs_emb = self.sub_emb(sub_xs)
        aa_xs_emb = self.aa_emb(aa_xs)

        # HierarchicalEGNN message passing


        # Readout of node features for each graph level 
        
