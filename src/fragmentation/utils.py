from rdkit import Chem
from rdkit.Chem import Mol, Atom, BondType
from transform.base import SMILES

def create_bond(rwmol, idx1, idx2, bondtype):
    rwmol.AddBond(idx1, idx2, bondtype)
    for idx in [idx1, idx2]:
        atom = rwmol.GetAtomWithIdx(idx)
        atom_numexplicitHs = atom.GetNumExplicitHs()
        if atom_numexplicitHs:
            atom.SetNumExplicitHs(atom_numexplicitHs - 1)

def add_dummy_atom(rwmol, index, bondtype=BondType.SINGLE, label=0):
    dummy_atom = Atom("*")
    dummy_atom.SetIsotope(label)  
    new_idx = rwmol.AddAtom(dummy_atom)
    create_bond(rwmol, index, new_idx, bondtype)

def convert_to_rdmol(mol: SMILES|Mol, 
                     isomericSmiles: bool = True
                     ):
    if isomericSmiles :
        if isinstance(mol, SMILES) :
            mol = Chem.MolFromSmiles(mol)
    else :
        smiles = convert_to_SMILES(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smiles)
    return mol

def convert_to_SMILES(mol: SMILES|Mol, canonicalize: bool = False,
                      isomericSmiles: bool = True) -> SMILES:
    if isomericSmiles :
        if isinstance(mol, Mol) :
            mol = Chem.MolToSmiles(mol)
        elif canonicalize is True :
            mol = Chem.MolToSmiles(Chem.MolFromSmiles(mol))
    else :
        if isinstance(mol, Mol) :
            mol = Chem.MolToSmiles(mol, isomericSmiles=False)
        else :
            mol = Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=False)
    return mol

