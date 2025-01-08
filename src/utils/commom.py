from rdkit import Chem
from rdkit.Chem import Mol
from .typing import SMILES

def convert2rdmol(mol: SMILES|Mol,
                  isomericSmiles: bool=False
                  ):
    if isomericSmiles:
        if isinstance(mol, SMILES):
            mol = Chem.MolFromSmarts(mol)
    else:
        smiles = convert2SMILES(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smiles)
    return mol

def convert2SMILES(mol: SMILES|Mol,
                   canonicalize: bool=False,
                   isomericSmiles: bool=True,
                   ):
    if isomericSmiles:
        if isinstance(mol, Mol):
            mol = Chem.MolToSmiles(mol)
        elif canonicalize is True:
            mol = Chem.MolToSmiles(Chem.MolFromSmiles(mol))
    else:
        if isinstance(mol, Mol):
            mol = Chem.MolToSmiles(mol, isomericSmiles=False)
        else:
            mol = Chem.MolToSmiles(Chem.MolFromSmiles(mol), \
                                   isomericSmiles=False)
    return mol

def check_and_convert2rdmol(mol: SMILES|Mol,
                            isomericSmiles = True
                            ):
    assert isinstance(mol, Mol) or isinstance(mol, SMILES)
    return convert2rdmol(mol, isomericSmiles)

def check_and_convert2SMILES(mol: SMILES|Mol,
                             canonicalize: bool=False,
                             isomericSmiles=True
                             ):
    assert isinstance(mol, Mol) or isinstance(mol, SMILES)
    return convert2SMILES(mol, canonicalize, isomericSmiles)