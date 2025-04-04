from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from rdkit.Chem.BRICS import BRICSDecompose
from itertools import cycle
import re

def remove_dummy(frag):
    query = Chem.MolFromSmiles('*')
    repl = Chem.MolFromSmiles("[H]")
    test_H = AllChem.ReplaceSubstructs(frag, query, repl, True)[0]
    
    return Chem.RemoveHs(test_H)

def split_submol(mol, bond):
    bond_id = bond.GetIdx()
    fragmented_mol = rdmolops.FragmentOnBonds(mol, [bond_id], addDummies=True)
    fragmented_mol = Chem.GetMolFrags(fragmented_mol, asMols=True)

    return fragmented_mol

def remove_radicals(mol):
    """ Remove unwanted radicals """
    rwmol = Chem.RWMol(mol)

    for atom in rwmol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)

    mol_fixed = rwmol.GetMol()
    return mol_fixed

def has_ring(mol):
    return len([list(ring) for ring in Chem.GetSymmSSSR(mol)]) > 0

def get_rings(mol, ids_list):
    submols = []
    for ids in ids_list:
        bonds = set()

        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in ids and bond.GetEndAtomIdx() in ids:
                bonds.add(bond)
 
        # Scomponi ulteriormente gruppi ciclici
        cyclic_atoms = [bond.GetEndAtom() for bond in bonds]
        for atom in cyclic_atoms:
            for bond in atom.GetBonds():
                begin_id = bond.GetBeginAtomIdx()
                end_id = bond.GetEndAtomIdx()
                if end_id not in sum(ids_list, []) or begin_id not in sum(ids_list, []):

                    external_atom = mol.GetAtomWithIdx(end_id) if end_id not in sum(ids_list, []) else mol.GetAtomWithIdx(begin_id)
                    if len(external_atom.GetNeighbors()) > 1:
                        submols += split_submol(mol, bond)
                    else:
                        bonds.add(bond)
        
        bonds = list(set(map(lambda b: b.GetIdx(), bonds)))
        submols.append(Chem.PathToSubmol(mol, bonds))
        
    return submols 

def decompose_molecule(molecule):
    mols = BRICSDecompose(molecule, returnMols=True)

    raw_submol = []
    for mol in mols: 
        ring_atom_ids = [list(ring) for ring in Chem.GetSymmSSSR(mol)]
        if len(ring_atom_ids) > 0:
            raw_submol += get_rings(mol, ring_atom_ids)
        else:
            raw_submol.append(mol)

    # Remove duplicates and multiple rings submolecules 
    processed_submols = []
    processed_smiles = []
    for submol in raw_submol:
        submol_smiles = Chem.MolToSmiles(submol) 
        submol = Chem.MolFromSmiles(submol_smiles)

        if not submol or submol.GetRingInfo().NumRings() > 1:
            continue
        
        processed = remove_dummy(submol)
        processed = remove_radicals(processed)
        if Chem.MolToSmiles(processed) not in processed_smiles:
            processed_submols.append(processed)
            processed_smiles.append(Chem.MolToSmiles(processed))

    return processed_smiles
