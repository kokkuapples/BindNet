import os 
import csv
import numpy as np
from itertools import cycle
import sys
from collections import defaultdict
from rdkit import Chem
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
import torch
from pathlib import Path
from Bio import PDB
import tqdm
import re

from features_extractor import FeatureExtractor, process_key, dist_filter

class HierarchicalProteinGraph(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "g2a_index":
            return torch.tensor([[self.subgroup_nodes.size(0)], [self.atom_nodes.size(0)]])
        if key == "aa2g_index":
            return torch.tensor([[self.aa_nodes.size(0)], [self.subgroup_nodes.size(0)]])
        if key == "atom_edge_index":
            return self.atom_nodes.size(0)  
        if key == "subgroup_edge_index":
            return self.subgroup_nodes.size(0)  
        if key == "aa_edge_index":
            return self.aa_nodes.size(0)  

        return super().__inc__(key, value, *args, **kwargs)

class HierarchicalProteinExtractor(FeatureExtractor):
    def __init__(self, molecule, pdb_file, dropHs:bool=True, cut_dist=5.0, node_num=None):
        super().__init__(molecule, dropHs, cut_dist, node_num)
        self.pdb_file = pdb_file

        self.__subgroup_encoding = self.__read_encoding("subgroups_encoding.csv")
        self.__aa_encoding = self.__read_encoding("aminoacids_encoding.csv")

    def build_subgroup_graph(self):
        # Match subgroups to protein and get a2g and g2a representation
        matches = self.__get_g2a_repr()
        self.__g2a_repr, subgroup_smiles = self.__detect_collision(matches)
        self.__get_a2g_repr()

        # Get tensor of tokenized subgroups
        subgroups_xs = torch.tensor([self.__subgroup_encoding[subgroup] for subgroup in subgroup_smiles])
        
        # Get subgroup -> subgroup edges
        coords = [self.__compute_center_of_mass(ids) for ids in self.__g2a_repr.values()]
        subgroup_edges_index = self.__topological_based_matching(self.__g2a_repr)
        # Get subgroup -> atom edges
        subgroup_atom_edges_index = self.__get_subgroup_atom_edges()

        return subgroups_xs, subgroup_edges_index, subgroup_atom_edges_index, coords

    def build_aminoacid_graph(self):
        # The AA graph needs information from subgroup level, if the latter is not created then create it
        if not hasattr(self, "__a2g_repr"):
            self.build_subgroup_graph()

        mapping = self.__pdb2rdkit_mapping()

        # Get tensor of AA chain tokenized
        aa_xs = self.__get_AA_tokens(mapping)

        # Get AA -> AA edges
        self.__get_AA2a_repr(mapping)
        coords = [self.__compute_center_of_mass(ids) for ids in self.__AA2a_repr.values()]
        aa_edges_index = self.__topological_based_matching(self.__AA2a_repr)

        # Get AA -> subgrup edges
        aa_subgroup_edges_index = self.__get_AA_subgroup_edges()
        
        return aa_xs, aa_edges_index, aa_subgroup_edges_index, coords

    def __get_AA_tokens(self, mapping):
        aa_list = list(set(mapping.values()))
        aa_list.sort(key=lambda t: t[1])
        
        aa_xs = torch.tensor([self.__aa_encoding[val[0]] for val in aa_list])

        return aa_xs

    def __get_subgroup_atom_edges(self):
        subgroup_index = []
        atom_index = [] 
        for atom_id in self.__a2g_repr.keys():
                for subgroup in self.__a2g_repr[atom_id]:
                    # Edge (i, j): i -> j
                    subgroup_index.append(subgroup)
                    atom_index.append(atom_id)

        edges_index = torch.tensor([subgroup_index, atom_index])
        return edges_index

    def __get_AA_subgroup_edges(self):
        submols_index = []
        aas_index = []
        edges_set = set()
        for aa, atoms in self.__AA2a_repr.items():
            for atom in atoms:
                for submol in self.__a2g_repr[atom]:
                    edges_set.add((aa, submol))
                    submols_index.append(submol)
                    aas_index.append(aa)
        
        aas_index = [hyper_edge[0] for hyper_edge in edges_set]
        submols_index = [hyper_edge[1] for hyper_edge in edges_set]
        edges_index = torch.tensor([aas_index, submols_index])

        return edges_index

    def __detect_collision(self, matches):
        index = iter(range(self.molecule.GetNumAtoms()))
        total_covered = set()
        filtered_matches = dict()
        subgroups_list = []

        matches = sorted(matches, key=lambda t: len(t[0]), reverse=True)
        for query, subgroup_smiles in matches:
            if set(query).difference(total_covered):
                filtered_matches[next(index)] = query
                total_covered.update(set(query))
                subgroups_list.append(subgroup_smiles)

        return filtered_matches, subgroups_list

    def __get_g2a_repr(self):
        matches = []

        Chem.Kekulize(self.molecule, clearAromaticFlags=True)
        for subgroup_smiles in self.__subgroup_encoding.keys(): # Already in SMILES
            frag_for_match = Chem.MolFromSmiles(re.sub(r"=", r"~", subgroup_smiles))
            
            matched_atoms = self.molecule.GetSubstructMatches(frag_for_match)
            if matched_atoms:
                # (atom_ids (tuple), subgroup_smiles)
                matches += list(zip(matched_atoms, cycle([subgroup_smiles])))
        
        return matches

    def __get_a2g_repr(self):
        self.__a2g_repr = defaultdict(list)
        for group, atoms in self.__g2a_repr.items():
            for atom in atoms:
                self.__a2g_repr[atom].append(group)

    def __get_AA2a_repr(self, mapping):
        self.__AA2a_repr = dict()
        aa_ids = {aa_id for (aa_name, aa_id) in mapping.values()}
        
        for aa_target in aa_ids:
            selected_atoms = [atom for atom, (aa_name, aa_id) in mapping.items() if aa_id == aa_target]
            self.__AA2a_repr[aa_target] = selected_atoms

    def __parse_pdb_residues(self):
        parser = PDB.PDBParser(QUIET=True)
        pdb_protein  = parser.get_structure("structure", self.pdb_file)

        pdb_atoms = []
        for model in pdb_protein:
            for chain in model:
                for residue in chain:
                    if PDB.is_aa(residue, standard=True):
                        res_id = residue.get_id()[1]
                        res_name = residue.get_resname()

                        for atom in residue:
                            atom_coord = atom.get_coord()

                            # Format: (atom coordinates (x, y, z), residue name (e.g. "ALA", "ARG"), residue id)
                            pdb_atoms.append((np.array(atom_coord), res_name, res_id))
        return pdb_atoms

    def __pdb2rdkit_mapping(self):
        pdb_atoms = self.__parse_pdb_residues()
        rdkit_atoms = self.molecule.GetAtoms()
        atoms_mapping = dict()

        for i in range(len(rdkit_atoms)):
            rdkit_coord = self.molecule.GetConformer().GetAtomPosition(i)
            coord = np.array([rdkit_coord.x, rdkit_coord.y, rdkit_coord.z])

            # Use spatial-based mapping
            for pdb_coord, res_name, res_id in pdb_atoms:
                if np.linalg.norm(pdb_coord - coord) < 0.1:
                    atoms_mapping[i] = (res_name, res_id)
                    continue

        # Remap AA id starting from zero
        offset_id = min(atoms_mapping.values(), key=lambda t: t[1])[1]
        atoms_mapping = {rdkit_atom_id: (res_name, res_id - offset_id) for rdkit_atom_id, (res_name, res_id) in atoms_mapping.items()}
        return atoms_mapping
    
    def __read_encoding(self, filename, base_dir="encodings", delimiter=","):   
        filepath = os.path.join(".", base_dir, filename)

        with open(filepath) as csv_file:
            reader = csv.reader(csv_file, delimiter=delimiter)
            next(reader) # Skip header row
            embedding_dict = {row[0]: int(row[1]) for row in reader}

        return embedding_dict

    def __compute_center_of_mass(self, atom_ids):
        coords = []
        weights = []
        
        for atom_id in atom_ids:
            rdkit_coord = self.molecule.GetConformer().GetAtomPosition(atom_id)
            coords.append([rdkit_coord.x, rdkit_coord.y, rdkit_coord.z])
            weights.append(self.molecule.GetAtomWithIdx(atom_id).GetMass())

        com_coords = np.average(coords, axis=0, weights=weights)
        return com_coords

    def __distance_based_matching(self, repr_, edge_threshold) :
        center_of_masses = np.array([self.__compute_center_of_mass(atoms) for atoms in repr_.values()]) 
        
        index1 = []
        index2 = []
        for i in range(center_of_masses.shape[0]):
            for j in range(i+1, center_of_masses.shape[0]):
                if np.linalg.norm(center_of_masses[i] - center_of_masses[j]) < edge_threshold:
                    index1 += [i, j]
                    index2 += [j, i]

        edges_index = torch.tensor([index1, index2])
        return edges_index

    def __topological_based_matching(self, repr_):
        substructures = repr_.keys()
        edges = set()
        for i, substruct1 in enumerate(substructures):
            for j, substruct2 in enumerate(substructures):
                if i != j:
                    # TODO: Find neighbors using atom-node edges (distance based)
                    neighbors = [list(self.molecule.GetAtomWithIdx(atom).GetNeighbors()) for atom in repr_[substruct1]]
                   
                    # Flat the list and get atom idx
                    neightbors = sum(neighbors, [])
                    neighbors_index = set([a.GetIdx() for a in neightbors])

                    substructure_to_atom = repr_[substruct2]
                    if neighbors_index.intersection(substructure_to_atom):
                        edges.add((substruct1, substruct2))
                        edges.add((substruct2, substruct1))
        
        substruct1_indexes = [pair[0] for pair in edges]
        substruct2_indexes = [pair[1] for pair in edges]
        edges_tensor = torch.tensor([substruct1_indexes, substruct2_indexes])

        return edges_tensor

def __get_ligand_pocket_bonds(pocket_coord, ligand_coord, offset):
    dm = cdist(pocket_coord, ligand_coord)
    pocket_idx, ligand_idx = dist_filter(dm, 12)
    
    ligand_idx += offset
    pocket_ligand_edges = np.vstack([np.hstack([pocket_idx, ligand_idx]), 
                                    np.hstack([ligand_idx, pocket_idx])])

    return torch.tensor(pocket_ligand_edges)

def __convert_to_tensor(edges):
    node1 = [edge[0] for edge in edges]
    node2 = [edge[1] for edge in edges]

    return torch.tensor([node1, node2])

def __process_pair(pocket_fe, ligand_fe):
    # Atom pocket-ligand graph processing
    # TODO: Check for bidirectional edges LP
    atom_xs, offset_atom, atom_edges_index_p, atom_edges_index_l, atom_pl_edges, atom_edge_features = process_key(pocket_fe, ligand_fe)
    
    # Convert (node, node) list into (2, N) tensors and then cat in one single tensor
    # Adding offset
    atom_edges_index_l = __convert_to_tensor(atom_edges_index_l) + offset_atom
    atom_edges_index_p = __convert_to_tensor(atom_edges_index_p)
    atom_pl_edges = __convert_to_tensor(atom_pl_edges)
    atom_edges_index = torch.cat([atom_edges_index_p, atom_edges_index_l, atom_pl_edges], dim=1)

    ### Subgroup pocket-ligand graph processing
    subgroup_xs_p, subgroup_edges_index_p, subgroup_atom_edges_index_p, subgroup_coord_p = pocket_fe.build_subgroup_graph()
    subgroup_xs_l, subgroup_edges_index_l, subgroup_atom_edges_index_l, subgroup_coord_l = ligand_fe.build_subgroup_graph()
    
    # Create a sigle tensor using an offset to distinguish pocket nodes from ligand nodes
    offset_subgroup = subgroup_xs_p.shape[0]
    subgroup_edges_index_l += offset_subgroup 

    subgroup_atom_edges_index_l[0,] += offset_subgroup
    subgroup_atom_edges_index_l[1,] += offset_atom
    
    subgroup_xs = torch.cat([subgroup_xs_p, subgroup_xs_l])
    subgroup_atom_edges_index = torch.cat([subgroup_atom_edges_index_p, subgroup_atom_edges_index_l], dim=1)
    
    subgroup_pl_edges = __get_ligand_pocket_bonds(subgroup_coord_p, subgroup_coord_l, offset_subgroup)
    subgroup_edges_index = torch.cat([subgroup_edges_index_p, subgroup_edges_index_l, subgroup_pl_edges], dim=1)
    subgroup_edge_features = torch.cat([torch.ones(subgroup_edges_index_p.shape[1]),
                                        torch.ones(subgroup_edges_index_l.shape[1]),
                                        torch.ones(subgroup_pl_edges.shape)[1] * -1])
    
    ### Aminoacid pocket-ligand graph processing
    aa_xs_p, aa_edges_index_p, aa_subgroup_edges_index_p, aa_coord_p = pocket_fe.build_aminoacid_graph()

    # In AA level the ligand is a single node
    aa_xs_l = torch.tensor([99])
    aa_edges_index_l = torch.tensor([])
    aa_subgroup_edges_index_l = __convert_to_tensor([(99, a) for a in subgroup_xs_l])
    aa_coord_l = np.expand_dims(np.mean(subgroup_coord_l, axis=0), axis=0)
    #aa_xs_l, aa_edges_index_l, aa_subgroup_edges_index_l, aa_cord_l = ligand_fe.build_aminoacid_graph()

    offset_aa = aa_edges_index_p.shape[0]
    aa_edges_index_l += offset_aa

    aa_subgroup_edges_index_l[0,] += offset_aa
    aa_subgroup_edges_index_l[1,] += offset_subgroup

    aa_xs = torch.cat([aa_xs_p, aa_xs_l])
    aa_subgroup_edges_index = torch.cat([aa_subgroup_edges_index_p, aa_subgroup_edges_index_l], dim=1)
    
    aa_pl_edges = __get_ligand_pocket_bonds(aa_coord_p, aa_coord_l, offset_aa)
    aa_edges_index = torch.cat([aa_edges_index_p, aa_edges_index_l, aa_pl_edges], dim=1)
    aa_edge_features = torch.cat([torch.ones(aa_edges_index_p.shape[1]),
                                  #torch.ones(aa_edges_index_l.shape[1]),
                                  torch.ones(aa_pl_edges.shape)[1] * -1])

    # Packaging different layers
    atom_graph = (atom_xs, atom_edges_index, None, atom_edge_features)
    subgroup_graph = (subgroup_xs, subgroup_edges_index, subgroup_atom_edges_index, subgroup_edge_features)
    aa_graph = (aa_xs, aa_edges_index, aa_subgroup_edges_index, aa_edge_features)

    return atom_graph, subgroup_graph, aa_graph 

def process_pdbbind(set_type, year="2016"):
    base_data_dir = os.path.join(os.getcwd(), "data", f"PDBbind{year}", set_type)

    # Check if the correct version of PDBbind is present
    assert os.path.isdir(base_data_dir), f"{base_data_dir} is not a valid path"

    index_name = f"INDEX_refined_data.{year}" if set_type == "refined-set" else f"INDEX_general_PL_data.{year}"
    index_file = os.path.join(base_data_dir, "index", f"INDEX_refined_data.{year}")

    # Read from index file labels and dirnames
    with open(index_file, "r") as f: 
        index_lines = [line.split() for line in f.readlines() if line[0] != "#"]

    pdb_names = [line[0] for line in index_lines] 
    labels = torch.Tensor([float(line[3]) for line in index_lines])
    pdb_names = pdb_names[:1]
    print(f"Processing {len(pdb_names)} proteins...")
    processed_data = []
    for label, pdb_filename in tqdm.tqdm(zip(labels, pdb_names)):
        pocket_file_path = Path(os.path.join(base_data_dir, f"{pdb_filename}/{pdb_filename}_pocket.pdb")) 
        ligand_file_path = Path(os.path.join(base_data_dir, f"{pdb_filename}/{pdb_filename}_ligand.mol2")) 
        
        pocket_fe = HierarchicalProteinExtractor.fromFile(pocket_file_path) 
        ligand_fe = HierarchicalProteinExtractor.fromFile(ligand_file_path) 

        atom_graph, subgroup_graph, aa_graph = __process_pair(pocket_fe, ligand_fe)
        atom_xs, atom_edges_index, _, atom_edge_features = atom_graph
        subgroup_xs, subgroup_edges_index, subgroup_atom_edges_index, subgroup_edge_features = subgroup_graph
        aa_xs, aa_edges_index, aa_subgroup_edges_index, aa_edge_features = aa_graph
        
        data = HierarchicalProteinGraph(atom_nodes=atom_xs, atom_edge_index=atom_edges_index, atom_edge_xs=atom_edge_features,
                     subgroup_nodes=subgroup_xs, subgroup_edge_index=subgroup_edges_index, g2a_index=subgroup_atom_edges_index, subgroup_edge_features=subgroup_edge_features,
                     aa_nodes=aa_xs, aa_edge_index=aa_edges_index, aa2g_index=aa_subgroup_edges_index, aa_edge_features=aa_edge_features, 
                     y=label)
        
        processed_data.append(data)
        print(data)
    
    # Check if processed dir exists otherwise create it
    save_path_dir = os.path.join(os.getcwd(), "data", f"PDBbind{year}", f"{set_type}_processed")
    Path(save_path_dir).mkdir(exist_ok=True)

    # Save processed data into .pt files
    for name, data in zip(pdb_names, processed_data):
        torch.save(data, os.path.join(save_path_dir, f"{name}_processed.pt")) 
    

if __name__ == "__main__":
    process_pdbbind("refined-set", year="2016")

