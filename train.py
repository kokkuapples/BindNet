import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import os

from hierarchical_graph import HierarchicalProteinGraph
from model import BindNet

SEED = 1337

def load_dataset(name, year, train_size=0.8, batch_size=1, shuffle=False):
    dir_path = os.path.join("data", f"PDBbind{year}", f"{name}_processed")
    loaded_data = [torch.load(os.path.join(dir_path, file)) for file in os.listdir(dir_path)]
    
    # Split and load data in batches
    train_data, val_data = train_test_split(loaded_data, train_size=train_size, random_state=SEED, shuffle=shuffle)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader 


if __name__ == "__main__":
    torch.manual_seed(SEED)

    train_loaded, val_loaded = load_dataset("refined-set", "2016", batch_size=1)
    # TODO: Load test set
    model = BindNet(input_dim=34, atom_edge_dim=9, aa_emb_dim=64, subgroup_emb_dim=64, hidden_dim=128, 
                    upper_edge_dim=2, edge_dim=32, n_layers=3, pdrop=0.0)
    optimizer = Adam(model.parameters())

    model.train()
    for batch in train_loaded:
        model.zero_grad()
        y_hat = model(batch)
        loss = F.mse_loss(y_hat, batch.y)
        loss.backward()
