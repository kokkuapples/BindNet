import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import os

from hierarchical_graph import HierarchicalProteinGraph
from model import BindNet

import lightning as L

SEED = 1337


class LitBindNet(L.LightningModule):
    def __init__(self, model, batch_size, shuffle, data_path, train_size):
        super().__init__()
        self.model = model

        ### Data loading parameters ###
        self.data_path = data_path
        self.train_size = train_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    def training_step(self, batch, batch_idx):
        y_hat = model(batch)
        loss = F.mse_loss(y_hat, batch.y)
        
        return loss
    
    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=1e-4)

    def train_dataloader(self):
        processed_path = os.path.join(self.data_path, "refined-set_processed")
        loaded_data = [torch.load(os.path.join(processed_path, file)) for file in os.listdir(processed_path)]
        
        train_data, val_data = train_test_split(loaded_data, train_size=self.train_size, 
                                                random_state=SEED, shuffle=self.shuffle)
   
        return DataLoader(train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        processed_path = os.path.join(self.data_path, "refined-set_processed")
        loaded_data = [torch.load(os.path.join(processed_path, file)) for file in os.listdir(processed_path)]
        
        train_data, val_data = train_test_split(loaded_data, train_size=self.train_size,
                                                random_state=SEED, shuffle=self.shuffle)
   
        return DataLoader(val_data, batch_size=self.batch_size)
    
    def test_dataloader(self):
        pass


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
    model = BindNet(input_dim=34, atom_edge_dim=9, aa_emb_dim=64, subgroup_emb_dim=64, hidden_dim=128, 
                    upper_edge_dim=2, edge_dim=32, n_layers=3, pdrop=0.0)
    
    # model, batch_size, shuffle, data_path, train_size
    data_path = os.pathl.join("data", "PDBbind2016")
    lit_model = LitBindNet(model, model, 32, True, data_path, 0.8)
    trainer = L.Trainer()
    trainer.fit(lit_model)
