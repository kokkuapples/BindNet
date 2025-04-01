import torch.nn as nn
import torch.functional as F
from egnn_pytorch import EGNN_Sparse
import torch


class HierarchicalEGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        
        self.egnn_list0 = nn.ModuleList([EGNN_Sparse(feats_dim=node_dim, edge_attr_dim=edge_dim) for _ in range(n_layers)])
        self.egnn_list1 = nn.ModuleList([EGNN_Sparse(feats_dim=node_dim, edge_attr_dim=edge_dim) for _ in range(n_layers)])
        self.egnn_list2 = nn.ModuleList([EGNN_Sparse(feats_dim=node_dim, edge_attr_dim=edge_dim) for _ in range(n_layers)])

    def stack_graphs(self, graph_u, xs_l):
        """ Aggregate in tensors for EGNN layer the upper and lower graph """

        xs_u, edge_index_u, bipartite_edge_index_u, edge_xs_u, coords_u = graph_u

        edge_index_u += xs_l.size(0)
        bipartite_edge_index_u[1,:] = xs_l.size(0)
        
        # Handle atom graph which has no lower layer
        if xs_l.numel() != 0:
            node_xs = torch.vstack([xs_l, torch.cat([coords_u, xs_u], dim=1)])
            edge_index = torch.hstack([edge_index_u, bipartite_edge_index_u]) 
            padding_size = (bipartite_edge_index_u.size(1), edge_xs_u.size(1))
            edge_xs = torch.vstack([edge_xs_u, torch.zeros(padding_size)])
        else:
            node_xs = torch.cat([coords_u, xs_u], dim=1)
            edge_index = edge_index_u 
            edge_xs = edge_xs_u
        
        return node_xs, edge_index, edge_xs

    def forward(self, atom_graph, subgroup_graph, aa_graph, batch):
        # Graph tuple: (xs, edge_index, bipartite_edge_index, edge_xs, coords)

        atom_xs, edges_index, edge_xs = self.stack_graphs(atom_graph, torch.Tensor([]))
        for i in range(self.n_layers):
            atom_xs = self.egnn_list0[i](atom_xs, edges_index, edge_attr=edge_xs)

        subgroup_xs, edges_index, edge_xs = self.stack_graphs(subgroup_graph, atom_xs)
        for i in range(self.n_layers):
            subgroup_xs = self.egnn_list1[i](subgroup_xs, edges_index, edge_attr=edge_xs)

        subgroup_xs = subgroup_xs[:subgroup_graph[4].size(0),:]
        aa_xs, edges_index, edge_xs = self.stack_graphs(aa_graph, subgroup_xs)

        for i in range(self.n_layers):
            aa_xs = self.egnn_list2[i](aa_xs, edges_index, edge_attr=edge_xs)

        # Return only transformed features not coords
        return atom_xs[:, 3:], subgroup_xs[:, 3:], aa_xs[:, 3:]

class BindNet(nn.Module):
    def __init__(self, input_dim, atom_edge_dim, aa_emb_dim, subgroup_emb_dim, 
                    upper_edge_dim, hidden_dim, edge_dim, n_layers, pdrop=0.0):

        super().__init__()
        self.n_layers = n_layers

        self.atom_proj = nn.Linear(input_dim, hidden_dim)
        self.sub_emb = nn.Embedding(subgroup_emb_dim, hidden_dim)
        self.aa_emb = nn.Embedding(aa_emb_dim, hidden_dim)

        self.atom_edge_proj = nn.Linear(atom_edge_dim, edge_dim)
        self.upper_edge_proj = nn.Linear(upper_edge_dim, edge_dim)
        
        self.hegnn = HierarchicalEGNN(hidden_dim, edge_dim, n_layers)
        self.mlp = nn.Sequential(nn.Linear(3*hidden_dim, hidden_dim),
                                 nn.Dropout(pdrop),
                                 nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.Dropout(pdrop),
                                 nn.SiLU(),
                                 nn.Linear(hidden_dim, 1)
                              )
        
    def forward(self, hg):
        # Unpacking hierarchical graph in atom, subgroup and aminoacid graphs
        atom_xs, atom_edges, atom_edges_xs, = hg.atom_nodes, hg.atom_edge_index, hg.atom_edge_xs
        sub_xs, sub_edges, g2a_edges, sub_edges_xs = hg.subgroup_nodes, hg.subgroup_edge_index, hg.g2a_index, hg.subgroup_edge_features
        aa_xs, aa_edges, aa2g_edges, aa_edges_xs = hg.aa_nodes, hg.aa_edge_index, hg.aa2g_index, hg.aa_edge_features
        
        # Get coords and features
        atom_coords, atom_features = atom_xs[:,:3], atom_xs[:, 3:]
        sub_coords, sub_features = sub_xs[:, :3], sub_xs[:, 3:].to(torch.int32)
        aa_coords, aa_features = aa_xs[:, :3], aa_xs[:, 3:].to(torch.int32)

        # Project atom features and get embeddings for subgroups and AAs
        atom_xs_proj = self.atom_proj(atom_features) # Tensor: (Na, hidden_dim)
        sub_xs_emb = self.sub_emb(sub_features).squeeze(dim=1)
        aa_xs_emb = self.aa_emb(aa_features).squeeze(dim=1)

        # Edge projection
        atom_edges_xs_proj = self.atom_edge_proj(atom_edges_xs)
        sub_edges_xs_proj = self.upper_edge_proj(sub_edges_xs)
        aa_edges_xs_proj = self.upper_edge_proj(aa_edges_xs)
        
        # HierarchicalEGNN message passing
        atom_graph = (atom_xs_proj, atom_edges, torch.tensor([[], []]), atom_edges_xs_proj, atom_coords)
        sub_graph = (sub_xs_emb, sub_edges, g2a_edges, sub_edges_xs_proj, sub_coords)
        aa_graph = (aa_xs_emb, aa_edges, aa2g_edges, aa_edges_xs_proj, aa_coords)
        
        atom_xs_out, sub_xs_out, aa_xs_out = self.hegnn(atom_graph, sub_graph, aa_graph, hg.ptr)

        # Readout of node features for each graph level
        readout = torch.cat([torch.mean(atom_xs_out, axis=0), # Tensor: (3 * hidden_dim)
                             torch.mean(sub_xs_out, axis=0), 
                             torch.mean(aa_xs_out, axis=0)])

        # Concat the graphs features and pass to an MLP
        y_hat = self.mlp(readout) 

        return y_hat
