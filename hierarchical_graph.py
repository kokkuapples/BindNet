import os 
import csv
import numpy as np
from itertools import cycle
from collections import defaultdict
from rdkit import Chem
from torch_geometric.data import Data
import torch
from pathlib import Path
from Bio import PDB
import re

from features_extractor import FeatureExtractor

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
        #subgroup_edges_index = self.__distance_based_matching(self.__g2a_repr, 3)
        subgroup_edges_index = self.__topological_based_matching(self.__g2a_repr)
        
        # Get subgroup -> atom edges
        subgroup_atom_edges_index = self.__get_subgroup_atom_edges()

        return subgroups_xs, subgroup_edges_index, subgroup_atom_edges_index 

    def build_aminoacid_graph(self):
        # The AA graph needs information from subgroup level, if the latter is not created then create it
        if not hasattr(self, "__a2g_repr"):
            self.build_subgroup_graph()

        mapping = self.__pdb2rdkit_mapping()

        # Get tensor of AA chain tokenized
        aa_xs = self.__get_AA_tokens(mapping)

        # Get AA -> AA edges
        self.__get_AA2a_repr(mapping)
        #aa_edges_index = self.__distance_based_matching(self.__AA2a_repr, 10) # tune the threshold
        aa_edges_index = self.__topological_based_matching(self.__AA2a_repr)

        # Get AA -> subgrup edges
        aa_subgroup_edges_index = self.__get_AA_subgroup_edges()
        
        return aa_xs, aa_edges_index, aa_subgroup_edges_index

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
                    neighbors = [list(self.molecule.GetAtomWithIdx(atom).GetNeighbors()) for atom in repr_[substruct1]]
                   
                    # Flat the list and get atom idx
                    neightbors = sum(neighbors, [])
                    neighbors_index = set([a.GetIdx() for a in neightbors])
                    #neighbors_index = set(map(lambda a: a.GetIdx(), neighbors))

                    substructure_to_atom = repr_[substruct2]
                    if neighbors_index.intersection(substructure_to_atom):
                        edges.add((substruct1, substruct2))
                        edges.add((substruct2, substruct1))
        
        substruct1_indexes = [pair[0] for pair in edges]
        substruct2_indexes = [pair[1] for pair in edges]
        edges_tensor = torch.tensor([substruct1_indexes, substruct2_indexes])

        return edges_tensor


if __name__ == "__main__":
    pdb_file_path = Path("pdb/1a1e_pocket.pdb") # Test protein pocket
    extractor = HierarchicalProteinExtractor.fromFile(pdb_file_path) 
    print("Creating graph")    
    subgroup_xs, subgroup_edges_index, subgroup_atom_edges_index = extractor.build_subgroup_graph()
    aa_xs, aa_edges_index, aa_subgroup_edges_index = extractor.build_aminoacid_graph()
