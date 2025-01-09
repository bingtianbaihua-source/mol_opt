from .fragmentation import Unit, Connection, FragmentedGraph
from rdkit import Chem
from .utils import add_dummy_atom, remove_bond
from rdkit.Chem import BRICS, Mol
from library import BlockLibrary

class BRICS_Connection(Connection):
    def __init__(self, 
                 unit1, 
                 unit2, 
                 atom_index1, 
                 atom_index2, 
                 brics_label1: str|int,
                 brics_label2: str|int,
                 bond_index, 
                 bondtype):
        super().__init__(unit1, unit2, atom_index1, atom_index2, bond_index, bondtype)
        self.brics_labels = (int(brics_label1), int(brics_label2))

class BRICS_Unit(Unit):
    def to_fragment(self, 
                    connection: BRICS_Connection):
        assert connection in self.connections
        atomMap = {}
        submol = self.graph.get_submol([self], \
                                       atomMap=atomMap)
        if self == connection.units[0]:
            atom_index = atomMap[connection.atom_indices[0]]
            brics_label = connection.brics_labels[0]
        else:
            atom_index = atomMap[connection.atom_indices[1]]
            brics_label = connection.brics_labels[1]
        bondtype = connection.bondtype

        rwmol = Chem.RWMol(submol)
        add_dummy_atom(rwmol, atom_index, bondtype, brics_label)
        fragment = rwmol.GetMol()
        Chem.SanitizeMol(fragment)
        return fragment

class BRICS_FragmentedGraph(FragmentedGraph):
    def fragmentation(self, mol: Mol):
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        rwmol = Chem.RWMol(mol)
        for (atom_idx1, atom_idx2), _ in brics_bonds:
            remove_bond(rwmol, atom_idx1, atom_idx2)
        broken_mol = rwmol.GetMol()

        atomMap = Chem.GetMolFrags(broken_mol)
        units = tuple(BRICS_Unit(self, atom_indices) for atom_indices in atomMap)

        unit_map = {}
        for unit in units:
            for atom_index in unit.atom_indices:
                unit_map[atom_index] = unit

        connections = []
        for brics_bond in brics_bonds:
            (atom_idx1, atom_idx2), (brics_label1, brics_label2) = brics_bond
            bond_index = mol.GetBondBetweenAtoms(atom_idx1, atom_idx2).GetIdx()
            bondtype = mol.GetBondBetweenAtoms(atom_idx1, atom_idx2).GetBondType()
            unit1 = unit_map[atom_idx1]
            unit2 = unit_map[atom_idx2] 
            connection = BRICS_Connection(unit1, 
                                          unit2, 
                                          atom_idx1, 
                                          atom_idx2, 
                                          brics_label1, 
                                          brics_label2, 
                                          bond_index, 
                                          bondtype)
            connections.append(connection)
        connections = tuple(connections)
        return units, connections
    
def brics_fragmentation(mol: Mol):
    return BRICS_FragmentedGraph(mol)

class BRICS_Fragmentation:
    fragmentation = staticmethod(brics_fragmentation)

class BRICS_BlockLibrary(BlockLibrary):
    fragmentation = BRICS_Fragmentation()

    @property
    def brics_labels_list(self):
        def get_brics_labels(rdmol: Mol):
            return str(rdmol.GetAtomWithIdx(0).GetIsotope())
        
        brics_labels: list[str] = [get_brics_labels(rdmol) for rdmol in self.rdmol_list]
        return brics_labels
