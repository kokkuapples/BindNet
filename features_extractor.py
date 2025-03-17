import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdchem
from collections import OrderedDict
from scipy.spatial.distance import cdist
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import scipy.spatial as ss
from scipy.spatial.distance import cdist
import pickle
import h5py
# log all the anomalies
import logging
logging.basicConfig(filename='features_extraction.log', level=logging.INFO)

prot_atom_ids = [6, 7, 8, 16]
drug_atom_ids = [6, 7, 8, 9, 15, 16, 17, 35, 53]
pair_ids = [(i, j) for i in prot_atom_ids for j in drug_atom_ids]
ptable = Chem.GetPeriodicTable()

metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                    + list(range(37, 51)) + list(range(55, 84))
                    + list(range(87, 104)))

# List of tuples (atomic_num, class_name) with atom types to encode.
atom_classes = [
    (5, 'B'), (6, 'C'), (7, 'N'), (8, 'O'), (15, 'P'),
    (16, 'S'), (34, 'Se'), ([9, 17, 35, 53], 'halogen'),
    (metals, 'metal')]

bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92},
          'C': {'H': 109, 'C': 154 , 'N': 147, 'O': 143, 'F': 135},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142}}

bonds2 = {'H': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000},
          'C': {'H': -1000, 'C': 134, 'N': 129, 'O': 120, 'F': -1000},
          'N': {'H': -1000, 'C': 129, 'N': 125, 'O': 121, 'F': -1000},
          'O': {'H': -1000, 'C': 120, 'N': 121, 'O': 121, 'F': -1000},
          'F': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000}}

bonds3 = {'H': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000},
          'C': {'H': -1000, 'C': 120, 'N': 116, 'O': 113, 'F': -1000},
          'N': {'H': -1000, 'C': 116, 'N': 110, 'O': -1000, 'F': -1000},
          'O': {'H': -1000, 'C': 113, 'N': -1000, 'O': -1000, 'F': -1000},
          'F': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000}}
margin1, margin2, margin3 = 10, 5, 3
bond_dict = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}



class FeatureExtractor:
    def __init__(self, molecule, pdb_file, dropHs:bool=True, cut_dist=5.0, node_num=None):
        """
        Contains all methods to extract features from a molecule.
        """
        self.molecule = molecule
        self.pdb_file = pdb_file
        self.dropHs = dropHs
        self.cut_dist = cut_dist
        self.node_num = node_num
        if self.node_num is not None: # TODO: no idea why this is here
            molecule = Chem.RWMol(self.molecule)
            # reverse the molecule to only keep the first node_num atoms
            for atom in reversed(self.molecule.GetAtoms()):
                if atom.GetIdx() >= self.node_num:
                    molecule.RemoveAtom(atom.GetIdx())
            self.molecule = molecule.GetMol()
        if dropHs:
            self.molecule = Chem.RemoveHs(self.molecule)
        self.coords = None
        self.atom_nums = np.array([atom.GetAtomicNum() for atom in self.molecule.GetAtoms()])
        # np.array([atom.GetAtomicNum() for atom in self.molecule.GetAtoms()])

        # ONE HOT ENCODING  
        self.ATOM_CODES = {}
        self.FEATURE_NAMES = []

        for code, (atom, name) in enumerate(atom_classes):
            if type(atom) is list:
                for a in atom:
                    self.ATOM_CODES[a] = code
            else:
                self.ATOM_CODES[atom] = code
            self.FEATURE_NAMES.append(name)

        self.NUM_ATOM_CLASSES = len(atom_classes)

        self.SMARTS = {
            'hydrophobic': '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
            'aromatic':    '[a]',
            'acceptor':    '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
            'donor':       '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
            'ring':        '[r]'
            }

    def encode_num(self, atomic_num:int):
        """Encode atom type with a binary vector. If atom type is not included in
        the `atom_classes`, its encoding is an all-zeros vector.

        Parameters
        ----------
        atomic_num: int Atomic number

        Returns
        -------
        encoding: np.ndarray Binary vector encoding atom type (one-hot or null).
        """

        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding

    def get_from_smarts(self):
        atom_props = np.zeros((self.molecule.GetNumAtoms(), len(self.SMARTS)))
        for i, (prop, smarts_mol) in enumerate(self.SMARTS.items()):
            pattern = Chem.MolFromSmarts(smarts_mol)
            matches = self.molecule.GetSubstructMatches(pattern)
            atom_props[np.array(matches).flatten().tolist(), i] = 1
        return atom_props

    def PartialSanitizeMolecule(self, molecule_name="molecule"):
        """
        Apply a partial sanitization to the molecule. Return True if successful, False otherwise.
        """
        try:
            Chem.SanitizeMol(self.molecule, sanitizeOps=Chem.SANITIZE_PROPERTIES | Chem.SANITIZE_SYMMRINGS)
            return True
        except Exception as e:
            print(f"Warning: partial sanitization of {molecule_name} failed. Error: {e}")
            logging.error(f"Partial sanitization of {molecule_name} failed. Error: {e}")
            return False
        
    
    def ExtractAtomFeatures(self, molcode=None):
        """
        Extract atom features from a molecule, INCLUDING hybridization, heavy degree,
        hetero degree, and partial charge using Gasteiger charges.

        Returns
        -------
        coords : np.ndarray, shape=(n_atoms, 3)
            Numpy array containing the atom coordinates.
        features : np.ndarray, shape=(n_atoms, n_features)
            Numpy array containing the atom features.
        
        """
        features = []
        coords = []
        # Calcola le cariche di Gasteiger per la molecola
        Chem.AllChem.ComputeGasteigerCharges(self.molecule)
        conf = self.molecule.GetConformer()
        
        for atom in self.molecule.GetAtoms():
            # Ottieni le coordinate dell'atomo
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])

            atomic_number = atom.GetAtomicNum()
            # Proprietà aggiuntive richieste
            hybridization = atom.GetHybridization()
            heavy_degree = atom.GetTotalDegree() - atom.GetTotalNumHs()
            hetero_degree = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() not in [1, 6])
            partial_charge = float(atom.GetProp('_GasteigerCharge'))
            
            atom_features = np.hstack([
                self.encode_num(atomic_number),
                [int(hybridization),
                 heavy_degree,
                 hetero_degree]
                 #partial_charge, # Problems...
                #]
            ])
            # atom_features = [
            #     atomic_number,
            #     atom.GetDegree(),
            #     int(atom.GetChiralTag()),
            #     atom.GetTotalNumHs(),
            #     atom.GetFormalCharge(),
            #     atom.GetImplicitValence(),
            #     atom.GetExplicitValence(),
            #     ptable.GetRvdw(atomic_number),
            #     ptable.GetAtomicWeight(atomic_number),
            # ]
            features.append(atom_features)
        
        features = np.array(features, dtype=np.float32) 
        coords = np.array(coords, dtype=np.float32)

        if molcode is not None: # -1 if protein, 1 if ligand
            features = np.hstack((features, 
                            molcode*np.ones((features.shape[0], 1), dtype=np.float32)))
        features = np.hstack((features, self.get_from_smarts()))
        
        if self.node_num is not None: # TODO: no idea why this is here
            
            if len(features) != self.node_num: return None, None
            #assert len(features) == self.node_num, "This should be already handled."
            features = features[:self.node_num]
            coords = coords[:self.node_num]

        self.coords = coords
        
        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')
        return coords, features

    def ExtractBondFeatures(self, sign_features=False, recompute_edges=False):
        """
        Extract bond features from a RDKit molecule.
        Parameters
        
        -------
        bond_features : np.ndarray
            Numpy array containing the bond features.

        1.begin_atom_idx: Descrizione: Indice dell'atomo iniziale del legame.
        
        2.end_atom_idx: Indice dell'atomo finale del legame.   
        3.bond_type: Tipo di legame (singolo, doppio, ecc.) 
            Valori:
            1.0: Legame singolo
            2.0: Legame doppio
            3.0: Legame triplo
            1.5: Legame aromatico   
        4.is_aromatic: indica se il legame è aromatico.
            Valori:
            0: Non aromatico
            1: Aromatico   
        5.is_in_ring: indica se il legame è in un anello.
            Valori:
            0: Non in anello
            1: In anello   
        6.stereo_value: Informazioni stereochimiche del legame.
            Valori:
            0: STEREONONE (Nessuna stereochimica)
            1: STEREOANY (Qualsiasi stereochimica)
            2: STEREOZ (Cis/Z)
            3: STEREOE (Trans/E)
            4: STEREOCIS (Cis)
            5: STEREOTRANS (Trans)   
            7.stereo_atom_1: Indice del primo atomo coinvolto nella stereochimica del legame.   
            8.stereo_atom_2:Indice del secondo atomo coinvolto nella stereochimica del legame.   
            9.is_conjugated: indica se il legame è coniugato.
        Valori:
            0: Non coniugato
            1: Coniugato
        """

        if recompute_edges:
            kd_tree = ss.KDTree(self.coords)
            edge_tuples = list(kd_tree.query_pairs(self.cut_dist))
            edge_index = np.array(edge_tuples).T
            edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
            # replace bonds in the molecule
            self.molecule = Chem.RWMol(self.molecule)
            for bond in self.molecule.GetBonds():
                self.molecule.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for i in range(edge_index.shape[1]):
                self.molecule.AddBond(int(edge_index[0, i]), int(edge_index[1, i]), order=Chem.rdchem.BondType.SINGLE)
            self.molecule = self.molecule.GetMol()
            self.PartialSanitizeMolecule()
        bond_features = []
        edges = []
        for bond in self.molecule.GetBonds():
            if self.node_num is not None and (bond.GetBeginAtomIdx() >= self.node_num or bond.GetEndAtomIdx() >= self.node_num):
                continue
            edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            if sign_features:
                bond_features.append(bond.GetBondTypeAsDouble())
            else:
                stereo = bond.GetStereo()
                stereo_map = {
                    Chem.rdchem.BondStereo.STEREONONE: 0,
                    Chem.rdchem.BondStereo.STEREOANY: 1,
                    Chem.rdchem.BondStereo.STEREOZ: 2,
                    Chem.rdchem.BondStereo.STEREOE: 3,
                    Chem.rdchem.BondStereo.STEREOCIS: 4,
                    Chem.rdchem.BondStereo.STEREOTRANS: 5
                }
                stereo_value = stereo_map.get(stereo, -1)
                stereo_atoms = bond.GetStereoAtoms()
                stereo_atoms_list = list(stereo_atoms) if stereo_atoms else [-1, -1]
                bond_features.append([
                    bond.GetBondTypeAsDouble()*2, # make all integer
                    bond.GetIsAromatic(),
                    bond.IsInRing(),
                    stereo_value,
                    stereo_atoms_list[0],  
                    stereo_atoms_list[1],  
                    bond.GetIsConjugated()
                ])    
        bond_features = np.array(bond_features, dtype=float)
        edge_index = np.array(edges).T
        edge_dist = np.linalg.norm(self.coords[edge_index[0]] - self.coords[edge_index[1]], axis=1).reshape(-1, 1)
        bond_features = np.hstack([bond_features, edge_dist])
        assert np.isnan(bond_features).sum() == 0
        return edges, bond_features
    
    @classmethod
    def fromFile(cls, mol_file, dropHs:bool=True, tried_formats=None):
        """
        Converts a file to an RDKit Mol object.
        """
        suffix = mol_file.suffix
        if suffix == '.pdb':
            pdb_file = str(mol_file)
            mol = Chem.MolFromPDBFile(str(mol_file))
            node_num = 0
            with open(mol_file) as f:
                for line in f:
                    if 'REMARK' in line:
                        break
                for line in f:
                    cont = line.split()
                    # break
                    if cont[0] == 'CONECT':
                        break
                    node_num += int(cont[-1] != 'H' and cont[0] == 'ATOM')
            
        elif suffix == '.mol2':
            pdb_file = None
            mol = Chem.MolFromMol2File(str(mol_file))
            with open(mol_file) as f:
                node_num = 0
                for line in f:
                    if '<TRIPOS>ATOM' in line:
                        break
                for line in f:
                    cont = line.split()
                    if '<TRIPOS>BOND' in line or cont[7] == 'HOH':
                        break
                    node_num += int(cont[5][0] != 'H')

        elif suffix == '.sdf':
            pdb_file = None
            mol = Chem.MolFromMolFile(str(mol_file))
            with open(mol_file) as f:
                node_num = 0
                for line in f:
                    if '$$$$' in line:
                        break
                    node_num += 1
        else:
            raise ValueError("Invalid file format. Choose 'pdb', 'mol2', or 'sdf'.")
        if mol is None:
            if tried_formats is None:
                formats = [suffix[1:]]
            else:
                formats = tried_formats + [suffix[1:]]
            alt_files = [f for f in mol_file.parent.glob(f"{mol_file.stem}.*") if f.suffix[1:] not in formats]
            if alt_files:
                return cls.fromFile(alt_files[0], dropHs=dropHs, tried_formats=tried_formats)
            else:
                print(f"[{mol_file.parent.name}] Could not parse {suffix.upper()} file.")
                logging.error(f"[{mol_file.parent.name}] Could not parse {suffix.upper()} file.")
                return None
        
        return cls(mol, pdb_file, dropHs=dropHs, node_num=node_num)

    
    def SaveMolecule(self, filename, file_format='pdb'):
        """
        Salva una molecola RDKit in un file SDF o PDB utilizzando solo RDKit.
        """
        output_path = f"{filename}.{file_format}"
        if file_format == 'sdf':
            Chem.MolToMolFile(self.molecule, output_path)    
        elif file_format == 'pdb':
            Chem.MolToPDBFile(self.molecule, output_path)    
            print(f"File saved in {output_path}")
        else:
            raise ValueError("Invalid file format. Choose 'sdf' or 'pdb'.")
    
    
    



def GetAtomResidueId(atom):
    info = atom.GetPDBResidueInfo()
    return (info.GetResidueNumber(), info.GetResidueName(), info.GetChainId())

def AtomListToSubMol(mol, atom_indices, includeConformer=True):
    """
    Create a submolecule from a list of atom indices.    
    Parameters
    
    """
    atom_indices = sorted(atom_indices)
    em = Chem.RWMol(Chem.Mol())
    conformer = mol.GetConformer()   
    for idx in atom_indices:
        em.AddAtom(mol.GetAtomWithIdx(idx))
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in atom_indices and bond.GetEndAtomIdx() in atom_indices:
            em.AddBond(atom_indices.index(bond.GetBeginAtomIdx()), atom_indices.index(bond.GetEndAtomIdx()), bond.GetBondType())
    if includeConformer:
        conf = Chem.Conformer(len(atom_indices))
        for i, idx in enumerate(atom_indices):
            pos = conformer.GetAtomPosition(idx)
            conf.SetAtomPosition(i, pos)
        em.AddConformer(conf)
    return em

def ExtractPocketAndLigand(mol, cutoff=12.0, expandResidues=True,
                           ligand_residue=None, ligand_residue_blacklist=None,
                           append_residues=None):
    """
    Function extracting a ligand (the largest HETATM residue) and the protein
    pocket within certain cutoff. The selection of pocket atoms can be expanded
    to contain whole residues. The single atom HETATM residues are attributed
    to pocket (metals and waters).

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
        Molecule with a protein ligand complex
    cutoff: float (default=12.0)
        Distance cutoff for the pocket atoms
    expandResidues: bool (default=True)
        Expand selection to whole residues within cutoff.
    ligand_residue: string (default None)
        Residue name which explicitly points to a ligand(s).
    ligand_residue_blacklist: array-like, optional (default None)
        List of residues to ignore during ligand lookup.
    append_residues: array-like, optional (default None)
        List of residues to append to pocket, even if they are HETATM, such
        as MSE, ATP, AMP, ADP, etc.
    
    """
    hetatm_residues = OrderedDict()
    protein_residues = OrderedDict()
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        res_id = GetAtomResidueId(atom)
        if info.GetIsHeteroAtom():
            if res_id not in hetatm_residues:
                hetatm_residues[res_id] = []
            hetatm_residues[res_id].append(atom.GetIdx())
        else:
            if res_id not in protein_residues:
                protein_residues[res_id] = []
            protein_residues[res_id].append(atom.GetIdx())
    if ligand_residue is not None and ligand_residue not in [res[1] for res in hetatm_residues]:
        raise ValueError('There is no residue named "%s" in the protein file' % ligand_residue)
    for res_id in list(hetatm_residues.keys()):
        if (len(hetatm_residues[res_id]) == 1 and res_id[1] != ligand_residue or
                append_residues is not None and res_id[1] in append_residues):
            protein_residues[res_id] = hetatm_residues[res_id]
            del hetatm_residues[res_id]
        elif ligand_residue is not None and res_id[1] != ligand_residue:
            del hetatm_residues[res_id]
        elif (ligand_residue_blacklist is not None and
              res_id[1] in ligand_residue_blacklist):
            del hetatm_residues[res_id]
    if len(hetatm_residues) == 0:
        raise ValueError('No ligands')
    ligand_key = sorted(hetatm_residues, key=lambda x: len(hetatm_residues[x]),
                        reverse=True)[0]
    ligand_amap = hetatm_residues[ligand_key]
    ligand = AtomListToSubMol(mol, ligand_amap, includeConformer=True)
    conf = ligand.GetConformer()
    ligand_coords = np.array([conf.GetAtomPosition(i)
                              for i in range(ligand.GetNumAtoms())])
    blacklist_ids = list(chain(*hetatm_residues.values()))
    protein_amap = np.array([i for i in range(mol.GetNumAtoms())
                             if i not in blacklist_ids])
    conf = mol.GetConformer()
    protein_coords = np.array([conf.GetAtomPosition(i)
                              for i in protein_amap.tolist()])
    mask = (cdist(protein_coords, ligand_coords) <= cutoff).any(axis=1)
    pocket_amap = protein_amap[np.where(mask)[0]].tolist()
    if expandResidues:
        pocket_residues = OrderedDict()
        for res_id in protein_residues.keys():
            if any(1 for res_aix in protein_residues[res_id]
                   if res_aix in pocket_amap):
                pocket_residues[res_id] = protein_residues[res_id]
        pocket_amap = list(chain(*pocket_residues.values()))
    pocket = AtomListToSubMol(mol, pocket_amap, includeConformer=True)
    protein = AtomListToSubMol(mol, protein_amap.tolist(), includeConformer=True)
    return protein, pocket, ligand


def dist_filter(dist_matrix, theta): 
    pos = np.where(dist_matrix<=theta)
    ligand_list, pocket_list = pos
    return ligand_list, pocket_list

def GetTypePair(pocket_fe, ligand_fe):
    assert pocket_fe.coords is not None and ligand_fe.coords is not None, "Extract atom features first."
    dm = cdist(pocket_fe.coords, ligand_fe.coords)
    pocks, ligs = dist_filter(dm, 12)
    
    bonds = np.concatenate([pocket_fe.atom_nums[pocks].reshape(-1, 1), ligand_fe.atom_nums[ligs].reshape(-1, 1)], axis=1)
    type_pair = [len(np.where((bonds == k).all(axis=1))[0]) for k in pair_ids]
    return type_pair
        
def load_pk_data(data_path):
    res = dict()
    with open(data_path) as f:
        for line in f:
            if '#' in line:
                continue
            cont = line.strip().split()
            if len(cont) < 5:
                continue
            code, pk = cont[0], cont[3]
            res[code] = float(pk)
    return res


def edge_ligand_pocket(dist_matrix, lig_size, theta=4, keep_pock=False, reset_idx=True):
    pos = np.where(dist_matrix<=theta)
    ligand_list, pocket_list = pos
    if keep_pock:
        node_list = range(dist_matrix.shape[1])
    else:
        node_list = sorted(list(set(pocket_list)))
    node_map = {node_list[i]:i+lig_size for i in range(len(node_list))}
    
    dist_list = dist_matrix[pos]
    if reset_idx:
        edge_list = [(x,node_map[y]) for x,y in zip(ligand_list, pocket_list)]
    else:
        edge_list = [(x,y) for x,y in zip(ligand_list, pocket_list)]
    
    edge_list += [(y,x) for x,y in edge_list]
    dist_list = np.concatenate([dist_list, dist_list])
    return dist_list, edge_list, node_map


def process_key(pocket_fe, ligand_fe, identity_features=True, keep_pock=False, theta=5.0):
    # Estrai la proteina intera, la tasca e il ligando
    # protein, pocket, ligand = ExtractPocketAndLigand(mol, cutoff=12.0, expandResidues=True)

    #Sanitizzazione parziale della molecola
    ligand_fe.PartialSanitizeMolecule(molecule_name="ligand")
    pocket_fe.PartialSanitizeMolecule(molecule_name="pocket")

    # esempio calcolo delle features
    pocket_coords, pocket_features = pocket_fe.ExtractAtomFeatures() 
    ligand_coords, ligand_features = ligand_fe.ExtractAtomFeatures()

    if pocket_features is None or ligand_features is None:
        return None

    # pocket_atom_nums = np.array([atom.GetAtomicNum() for atom in pocket.GetAtoms()])
    # ligand_atom_nums = np.array([atom.GetAtomicNum() for atom in ligand.GetAtoms()])

    type_pair = GetTypePair(pocket_fe, ligand_fe)

    # protein_bond_features = ExtractBondFeatures(protein)
    pocket_edges, pocket_bond_features = pocket_fe.ExtractBondFeatures()
    ligand_edges, ligand_bond_features = ligand_fe.ExtractBondFeatures()

    dm = cdist(ligand_coords, pocket_coords)
    lig_pock_dist, lig_pock_edge, node_map = edge_ligand_pocket(dm, ligand_features.shape[0], theta=theta, keep_pock=keep_pock)
    pocket_coords = pocket_coords[sorted(node_map.keys())]
    pocket_features = pocket_features[sorted(node_map.keys())]
    pockets_atoms = pocket_fe.atom_nums[sorted(node_map.keys())]
    # TODO: also pocket_edges would need to be updated accordingly

    if identity_features:
        node_features = np.vstack([
            np.hstack([ligand_features, np.zeros(ligand_features.shape)]),
            np.hstack([np.zeros(pocket_features.shape), pocket_features])
        ])
    else:
        node_features = np.vstack([ligand_features, pocket_features])

    edge_features = np.vstack([ligand_bond_features, pocket_bond_features])
    atoms_raw = np.concatenate([ligand_fe.atom_nums, pockets_atoms])
    coords = np.vstack([ligand_coords, pocket_coords])
    assert node_features.shape[0]==coords.shape[0]
    
    return node_features, ligand_features.shape[0], pocket_edges, ligand_edges, lig_pock_edge, edge_features
    #return {'separator': ligand_features.shape[0], 'coords': coords, 'node_features': node_features, 
    #        'edge_features': edge_features, 'atoms_raw': atoms_raw, 'type_pair': type_pair}


def process_pdbbind(data_dir:Path, dropHs:bool=True, test_k:str=None, year=2016):
    pk_dict = load_pk_data(data_dir/f'index/INDEX_general_PL_data.2020')
    if test_k is not None:
        keys_list = [test_k]
        train_keys = keys_list
        val_keys = []
        test_keys = []
    else:
        keys_list = [f.name for f in data_dir.iterdir() if f.is_dir() and len(f.name) == 4]
        assert set(keys_list) - set(pk_dict) == set(), "Some keys are missing."

        np.random.seed(42)         # train|val|test split: .8|.1|.1
        np.random.shuffle(keys_list)
        train_keys = keys_list[:int(.8*len(keys_list))]
        val_keys = keys_list[int(.8*len(keys_list)):int(.9*len(keys_list))]
        test_keys = keys_list[int(.9*len(keys_list)):]

    train_g = []
    val_g = []
    test_g = []
    for key_list, g in zip([train_keys, val_keys, test_keys], [train_g, val_g, test_g]):
        for k in tqdm(key_list):
            pocket_fe = FeatureExtractor.fromFile(data_dir/k/f"{k}_pocket.pdb", dropHs=dropHs)
            ligand_fe = FeatureExtractor.fromFile(data_dir/k/f"{k}_ligand.mol2", dropHs=dropHs)
            if pocket_fe is None or ligand_fe is None:
                continue

            result = process_key(pocket_fe, ligand_fe)
            if result:
                g.append(result)

    train_y = [pk_dict[k] for k in train_keys]
    val_y = [pk_dict[k] for k in val_keys]
    test_y = [pk_dict[k] for k in test_keys]
    return (train_g, train_y), (val_g, val_y), (test_g, test_y)


def get_bond_order(atom1, atom2, distance):
    distance = 100 * distance  # We change the metric

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < bonds1[atom1][atom2] + margin1:
        thr_bond2 = bonds2[atom1][atom2] + margin2
        if distance < thr_bond2:
            thr_bond3 = bonds3[atom1][atom2] + margin3
            if distance < thr_bond3:
                return 3
            return 2
        return 1
    return 0


def create_molecules(atom_symbols, coordinates, separator):
    """
    Create an RDKit molecule from atom symbols, 3D coordinates, and bond information.

    Parameters:
        atom_symbols (list of str): List of atomic symbols (e.g., ['C', 'O', 'H']).
        coordinates (list of tuple): List of (x, y, z) coordinates for each atom.
        bonds (list of tuple): Optional list of bonds. Each bond is a tuple of (atom_idx1, atom_idx2, bond_order).

    Returns:
        rdkit.Chem.Mol: The constructed RDKit molecule with 3D coordinates.
    """
    COVALENT_RADII = {
        "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
        "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39
    }
    # Validate input
    assert len(atom_symbols) == len(coordinates), "Number of atoms must match number of coordinates."

    protein = Chem.RWMol()
    ligand = Chem.RWMol()

    for i, symbol in enumerate(atom_symbols):
        atom = Chem.Atom(symbol)
        if i < separator:
            protein.AddAtom(atom)
        else:
            ligand.AddAtom(atom)

    # protein edges
    for i in range(separator):
        for j in range(i + 1, separator):
            p1 = coordinates[i]
            p2 = coordinates[j]
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            order = get_bond_order(atom_symbols[i], atom_symbols[j], dist)
            if order > 0:
                protein.AddBond(i, j, bond_dict[order])

    # ligand edges
    for i in range(separator, len(atom_symbols)):
        for j in range(i + 1, len(atom_symbols)):
            p1 = coordinates[i]
            p2 = coordinates[j]
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            order = get_bond_order(atom_symbols[i], atom_symbols[j], dist)
            if order > 0:
                ligand.AddBond(i - separator, j - separator, bond_dict[order])

    # Create conformer to store 3D coordinates
    conf_protein = Chem.Conformer(int(separator))
    for i, (x, y, z) in enumerate(coordinates[:separator]):
        conf_protein.SetAtomPosition(i, Point3D(x, y, z))
    protein.AddConformer(conf_protein)

    conf_ligand = Chem.Conformer(int(len(atom_symbols) - separator))
    for i, (x, y, z) in enumerate(coordinates[separator:]):
        conf_ligand.SetAtomPosition(i, Point3D(x, y, z))
    ligand.AddConformer(conf_ligand)

    # Return the finalized molecule
    return protein.GetMol(), ligand.GetMol()


def process_misato(data_dir:Path, dropHs:bool=True, year=2020):
    pk_dict = load_pk_data(data_dir/f'index/INDEX_general_PL_data.{year}')

    data_raw = h5py.File(data_dir, 'r')
    keys_list = list(data_raw.keys())
    np.random.seed(42)
    np.random.shuffle(keys_list)
    train_keys = keys_list[:int(.8*len(keys_list))]
    val_keys = keys_list[int(.8*len(keys_list)):int(.9*len(keys_list))]
    test_keys = keys_list[int(.9*len(keys_list)):]

    train_g = []
    val_g = []
    test_g = []
    for key_list, g in zip([train_keys, val_keys, test_keys], [train_g, val_g, test_g]):
        for k in key_list:
            coordinates = data_raw[k]['trajectory_coordinates'][0]
            atom_numbers = data_raw[k]['atoms_number'][:] # from number to symbol
            atom_symbols = [ptable.GetElementSymbol(int(i)) for i in atom_numbers]
            separator = separator = data_raw[k]["molecules_begin_atom_index"][:][-1]
            
            protein, ligand = create_molecules(atom_symbols, coordinates, separator)
            g.append(process_key(pocket, ligand, dropHs=dropHs))

    train_y = [pk_dict[k] for k in train_keys]
    val_y = [pk_dict[k] for k in val_keys]
    test_y = [pk_dict[k] for k in test_keys]
    return (train_g, train_y), (val_g, val_y), (test_g, test_y)


if __name__ == "__main__":
    dataset = 'pdbbind'
    raw_data_path = Path('data/PDBbind2020/refined-set')

    # dataset = 'misato'
    # raw_data_raw = Path('data/MD') / 'h5_files' / 'tiny_md.hdf5' # misato dataset
    year = '20' + raw_data_path.parent.name.split('_v')[-1] # TODO: Check this!!!!!!

    if dataset == 'pdbbind':
        train, val, test = process_pdbbind(raw_data_path, year=year)
    elif dataset == 'misato':
        train, val, test = process_misato(raw_data_path)

    with open(f'data/{dataset}{year}_train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(f'data/{dataset}{year}_val.pkl', 'wb') as f:
        pickle.dump(val, f)
    with open(f'data/{dataset}{year}_test.pkl', 'wb') as f:
        pickle.dump(test, f)
