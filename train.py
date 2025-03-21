import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import os

from hierarchical_graph import HierarchicalProteinGraph


def load_dataset(name, year, batch_size=1, shuffle=False):
    dir_path = os.path.join("data", f"PDBbind{year}", f"{name}_processed")
    loaded_data = [torch.load(os.path.join(dir_path, file)) for file in os.listdir(dir_path)]
    print(loaded_data[0])

    return DataLoader(loaded_data, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    seed = 1337
    torch.manual_seed(seed)

    train_loaded = load_dataset("refined-set", "2016")

        
