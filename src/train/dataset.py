from torch.utils.data import Dataset
from typing import List, Optional
from rdkit.Chem import Mol
from utils.typing import SMILES
from fragmentation import BR

class BBARDataset(Dataset):

    def __init__(self,
                 molecules: List[Mol|SMILES],
                 fragmented_molecules: Optional[List[]]):
        super().__init__()