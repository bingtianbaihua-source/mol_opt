from rdkit import Chem
from rdkit.Chem import Mol, Atom, BondType, RWMol
from src.utils import SMILES

def create_bond(rwmol, idx1, idx2, bondtype):
    rwmol.AddBond(idx1, idx2, bondtype)
    for idx in [idx1, idx2]:
        atom = rwmol.GetAtomWithIdx(idx)
        atom_numexplicitHs = atom.GetNumExplicitHs()
        if atom_numexplicitHs:
            atom.SetNumExplicitHs(atom_numexplicitHs - 1)

def remove_bond(rwmol: RWMol, 
                idx1, 
                idx2,
                ):
    rwmol.RemoveBond(idx1, idx2)
    for idx in [idx1, idx2]:
        atom = rwmol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
            atom.SetNumExplicitHs(1)

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

def check_dummy_atom(atom):
    return atom.GetAtomicNum() == 0

def find_dummy_atom(rdmol: Mol):
    for idx, atom in enumerate(rdmol.GetAtoms()):
        if check_dummy_atom(atom):
            return idx
    return idx

def get_dummy_bondtype(dummy_atom: Atom):
    bondtype = dummy_atom.GetTotalValence()
    assert bondtype in [1,2,3]
    if bondtype == 1:
        return BondType.SINGLE
    elif bondtype == 2:
        return BondType.DOUBLE
    else:
        return BondType.TRIPLE
    
def create_bond(rwmol: Mol,
                idx1,
                idx2,
                bondtype
                ):
    rwmol.AddBond(idx1, idx2, bondtype)
    for idx in [idx1, idx2]:
        atom = rwmol.GetAtomWithIdx(idx)
        atom_numexplicitHs = atom.GetNumExplicitHs()
        if atom_numexplicitHs:
            atom.SetNumExplicitHs(atom_numexplicitHs - 1)

def merge(scaffold: Mol,
          fragment: Mol,
          scaffold_index,
          fragment_index
          ):
    
    if fragment_index is None:
        fragment_index = find_dummy_atom(fragment)
    assert fragment_index is not None

    rwmol = Chem.RWMol(Chem.CombineMols(scaffold, fragment))
    dummy_atom_index = scaffold.GetNumAtoms() + fragment_index

    dummy_atom = rwmol.GetAtomWithIdx(dummy_atom_index)
    assert check_dummy_atom(dummy_atom)
    bondtype = get_dummy_bondtype(dummy_atom)

    fragment_index = dummy_atom.GetNeighbors()[0].GetIdx()
    create_bond(rwmol, scaffold_index, fragment_index, bondtype)
    rwmol.RemoveAtom(dummy_atom)
    mol = rwmol.GetMol()
    Chem.SanitizeMol(mol)

    return mol

