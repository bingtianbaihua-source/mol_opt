from torch.utils.data import Dataset
from typing import List, Optional
from rdkit.Chem import Mol
from utils.typing import SMILES
from fragmentation.brics import BRICS_FragmentedGraph, BRICS_BlockLibrary
from torch_geometric.data import Data as PyGData
from torch import FloatTensor
from transform.core import CoreGraphTransform

class BBARDataset(Dataset):

    def __init__(self,
                 molecules: List[Mol|SMILES],
                 fragmented_molecules: list[BRICS_FragmentedGraph],
                 properties: list[dict[str,float]],
                 library: BRICS_BlockLibrary,
                 library_pygdata_list: list[PyGData],
                 library_frequency: FloatTensor,
                 num_negative_samples: int,
                 train: bool = True,
                 ):
        super(BBARDataset, self).__init__()
        assert len(molecules) == len(properties)

        self.molecules = molecules
        self.properties = properties
        self.library = library
        self.library_pygdata = library_pygdata_list
        self.library_frequency = library_frequency
        self.num_negative_samples = num_negative_samples
        self.train = train

        self.core_transform = CoreGraphTransform.call
        