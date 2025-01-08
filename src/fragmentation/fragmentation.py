from typing import Tuple
from rdkit import Chem
from rdkit.Chem import Mol, BondType
from .utils import *
from transform.base import SMILES
from typing import List, Set
from itertools import combinations
import random

class Unit:
    def __init__(self,
                 graph, 
                 atom_indices: Tuple[int]
                 ):
        self.graph = graph
        atom_indices_set = set(atom_indices)
        bond_indices = []
        for bond in graph.rdmol.GetBonds():
            atom_idx1 = bond.GetBeginAtomIdx()
            atom_idx2 = bond.GetEndAtomIdx()
            if atom_idx1 in atom_indices_set and atom_idx2 in atom_indices_set:
                bond_indices.append(bond.GetIdx())

        self.atom_indices = atom_indices
        self.bond_indices = tuple(bond_indices)

        self.neighbors = []
        self.connections = []

    def add_connection(self,
                       neighbor_unit,
                       connection):
        self.neighbors.append(neighbor_unit)
        self.connections.append(connection)

    def to_rdmol(self):
        return self.graph.get_submol([self])
    
    def to_fragment(self, connection):
        assert connection in self.connections
        atomMap = {}
        submol = self.graph.get_submol([self], atomMap=atomMap)
        if self == connection.units[0]:
            atom_index = atomMap[connection.atom_indices[0]]
        else:
            atom_index = atomMap[connection.atom_indices[1]]
        bondtype = connection.bondtype

        rwmol = Chem.RWMol(submol)
        add_dummy_atom(rwmol, atom_index, bondtype)
        fragment = rwmol.GetMol()
        fragment = Chem.MolFromSmiles(Chem.MolToSmiles(fragment))
        return fragment
    
class Connection:
    def __init__(
            self,
            unit1: Unit,
            unit2: Unit,
            atom_index1: int,
            atom_index2: int,
            bond_index: int,
            bondtype: BondType
            ) -> None:
        self.units = (unit1, unit2)
        self.atom_indices = (atom_index1, atom_index2)
        self.bond_index = bond_index
        self._bondtype = int(bondtype)
        unit1.add_connection(unit2, self)
        unit2.add_connection(unit1, self)

    @property
    def bondtype(self):
        return BondType.values[self._bondtype]
    
class FragmentedGraph:
    def fragmentation(self, 
                      mol: Mol):
        raise NotImplementedError
    
    def __init__(self,
                 mol: SMILES|Mol) -> None:
        rdmol = convert_to_rdmol(mol, isomericSmiles=False)
        self.rdmol = rdmol

        units, connections = self.fragmentation(rdmol)
        self.units = units
        self.num_units = len(units)

        self.connections = connections
        self.connection_dict = {}
        for connection in connections:
            unit1, unit2 = connection.units
            self.connection_dict[(unit1, unit2)] = connection
            self.connection_dict[(unit2, unit1)] = connection

    def __len__(self):
        return self.num_units
    
    def get_submol(self,
                   unit_list: List[Unit],
                   atomMap: dict = {},
                   ):
        atom_indices = []
        bond_indices = []
        for unit in unit_list:
            atom_indices += unit.atom_indices
            bond_indices += unit.bond_indices
        for unit1,unit2 in combinations(unit_list, 2):
            connection = self.connection_dict.get((unit1, unit2))
            if connection is not None:
                bond_indices.append(connection.bond_index)

        atomMap.update({atom_index: new_atom_index for new_atom_index, atom_index in enumerate(atom_indices)})

        rwmol = Chem.RWMol()
        src_atom_list = [self.rdmol.GetAtomWithIdx(atom_index) \
                         for atom_index in atom_indices]
        src_bond_list = [self.rdmol.GetBondWithIdx(bond_index) \
                         for bond_index in bond_indices]
        
        for src_atom in src_atom_list:
            rwmol.AddAtom(src_atom)

        for src_bond in src_bond_list:
            src_atom_index1, src_atom_index2 = src_bond.GetBeginAtomIdx(), src_bond.GetEndAtomIdx()
            dst_atom_index1, dst_atom_index2 = atomMap[src_atom_index1], atomMap[src_atom_index2]
            bondtype = src_bond.GetBondType()
            rwmol.AddBond(dst_atom_index1, dst_atom_index2, bondtype)

        for src_atom, dst_atom in zip(src_atom_list, rwmol.GetAtoms()):
            if dst_atom.GetAtomicNum() == 7:
                degree_diff = src_atom.GetDegree() - dst_atom.GetDegree()
                if degree_diff > 0:
                    dst_atom.SetNumExplicitHs(dst_atom.GetNumExplicitHs() + degree_diff)

        submol = rwmol.GetMol()
        Chem.SanitizeMol(submol)
        return submol
    
    def get_datapoint(self,
                      traj=None):
        if traj is None:
            traj = self.get_subtrajectory(min_length=2)
        scaffold_units, fragment_unit = traj[:-1], traj[-1]

        if fragment_unit is None:
            scaffold = Chem.Mol(self.rdmol)
            return scaffold, None, (None, None)
        else:
            neighbor_units = set(fragment_unit.neighbors).intersection(set(scaffold_units))
            assert len(neighbor_units) == 1
            neighbor_unit = neighbor_units.pop()
            connection = self.connection_dict[(fragment_unit, neighbor_unit)]

            atomMap = {}
            scaffold = self.get_submol(scaffold_units, atomMap=atomMap)
            fragment = fragment_unit.to_fragment(connection)

            if fragment_unit is connection.units[0]:
                scaffold_atom_index = atomMap[connection.atom_indices[1]]
            else:
                scaffold_atom_index = atomMap[connection.atom_indices[0]]
            fragment_atom_index = fragment.GetNumAtoms() - 1  # atom index of dummy atom

            return scaffold, fragment, (scaffold_atom_index, fragment_atom_index)


    def get_subtrajectory(self,
                          length=None,
                          min_length=1,
                          max_length=None):
        if length is None:
            assert max_length is None or max_length >= min_length
            if max_length is None:
                max_length = self.num_units + 1
            else:
                max_length = min(max_length, self.num_units + 1)
            length = random.randrange(min_length, max_length+1)

        if length == self.num_units + 1:
            traj = list(self.units) + [None]
        else:
            traj: List[Unit] = []
            neighbors: Set[Unit] = set()
            traj_length = 0
            while True:
                if traj_length == 0:
                    unit = random.choice(self.units)
                else:
                    unit = random.choice(tuple(neighbors))
                traj.append(unit)
                traj_length += 1
                if traj_length == length:
                    break
                neighbors.update(unit.neighbors)
                neighbors = neighbors.difference(traj)
        return traj