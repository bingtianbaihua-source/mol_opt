from torch.utils.data import Dataset
from typing import List, Optional
from rdkit.Chem import Mol
from utils.typing import SMILES
from fragmentation.brics import BRICS_FragmentedGraph, BRICS_BlockLibrary
from torch_geometric.data import Data as PyGData
from torch import FloatTensor
import torch
from transform.core import CoreGraphTransform
import numpy as np

class MyDataset(Dataset):

    def __init__(self,
                 molecules: list[Mol|SMILES],
                 fragmented_molecules: list[BRICS_FragmentedGraph],
                 properties: list[dict[str,float]],
                 library: BRICS_BlockLibrary,
                 library_pygdata_list: list[PyGData],
                 library_frequency: FloatTensor,
                 num_negative_samples: int,
                 train: bool = True,
                 ):
        super(MyDataset, self).__init__()
        assert len(molecules) == len(properties)

        self.molecules = molecules
        self.properties = properties
        self.library = library
        self.library_pygdata = library_pygdata_list
        self.library_frequency = library_frequency
        self.num_negative_samples = num_negative_samples
        self.train = train

        self.core_transform = CoreGraphTransform.call

        if fragmented_molecules is not None:
            fragementation = self.library.fragmentation
            self.fragmented_molecules = [fragementation(mol) for mol in molecules]
        else:
            assert len(molecules) == len(fragmented_molecules)
            self.fragmented_molecules = fragmented_molecules

    def __len__(self):
        return len(self.fragmented_molecules)
    
    def __getitem__(self, index):
        core_rdmol, block_idx, core_atom_idx = self.get_datapoint(index)
        pygdata_core = self.core_transform(core_rdmol)
        condition: dict[str, float] = self.properties[index]

        num_core_atoms = core_rdmol.GetNumAtoms()
        y_atom = torch.full((num_core_atoms,), False, dtype=torch.bool)
        if block_idx is None:
            y_term = True
        else:
            y_term = False
            y_atom[core_atom_idx] = True

        pygdata_core.y_term = y_term
        pygdata_core.y_atom = y_atom

        if self.train:
            pos_pygdata : PyGData = None
            neg_pygdatas : list[PyGData] = None
            if block_idx is not None:
                pos_pygdata = self.library_pygdata[0]
                neg_pygdatas = [self.library_pygdata[0]] * self.num_negative_samples
            else:
                pos_pygdata = self.library_pygdata[block_idx]
                neg_idxs = self.get_negative_samples(block_idx)
                neg_pygdatas = [self.library_pygdata[neg_idx] for neg_idx in neg_idxs]
            return pygdata_core, condition, pos_pygdata, *neg_pygdatas
        else:
            if block_idx is None:
                pos_idx = 0
                neg_idxs = [0] * self.num_negative_samples
            else:
                pos_idx = block_idx
                neg_idxs = self.get_negative_samples(block_idx)
            return pygdata_core, condition, pos_idx, *neg_idxs

    def get_datapoint(self, index):
        fragmented_mol = self.fragmented_molecules[index]

        minlength = 2
        maxlength = len(fragmented_mol)+1
        length_list = np.arange(minlength, maxlength+1)
        length = np.random.choice(length_list, p=length_list / length_list.sum())
        traj = fragmented_mol.get_subtrajectory(length)

        datapoint = fragmented_mol.get_datapoint(traj)
        core_rdmol, block_rdmol, (core_atom_idx, _) = datapoint
        if block_rdmol is None:
            return (core_rdmol, None, None)
        block_index = self.library.get_index(block_rdmol)
        return (core_rdmol, block_index, core_atom_idx)

    def get_negative_samples(self, 
                             positive_sample: int):
        freq = torch.clone(self.library_frequency)
        freq[positive_sample] = 0.0
        neg_idxs = torch.multinomial(freq, self.num_negative_samples, True).tolist()
        return neg_idxs
