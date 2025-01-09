from .fragmentation import Fragmentation
from torch import FloatTensor
import os
import torch
import logging
from rdkit import Chem
from utils.typing import SMILES
from rdkit.Chem import Mol
from utils.commom import convert2SMILES

class BlockLibrary:
    fragmentation = Fragmentation()

    def __init__(self,
                 library_path: str|None,
                 smiles_list: list[str]|None,
                 frequency_list: FloatTensor|None,
                 use_frequency:bool = True,
                 save_rdmol: bool = False,
                 ):
        if library_path is not None:
            smiles_list, frequency_list = self.load_library_file(library_path, use_frequency)

        assert smiles_list is not None
        if not use_frequency:
            frequency_list = None

        self._smiles_list = smiles_list

        if frequency_list is not None:
            self._frequency_distribution = frequency_list
        else:
            if use_frequency:
                logging.warning('Frequency list is not provided. Using uniform distribution.')
            self._frequency_distribution = torch.full((len(smiles_list),), 1.0/len(smiles_list))

        self._smiles_to_index = {smiles: i for i, smiles in enumerate(smiles_list)}

        if save_rdmol:
            self._rdmol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        else:
            self._rdmol_list = None

    def __len__(self):
        return len(self._smiles_list)
    
    def __getitem__(self, index: int):
        return self._smiles_list[index]
    
    def get_rdmol(self, index: int):
        if self._rdmol_list is None:
            return Chem.MolFromSmiles(self._smiles_list[index])
        return self._rdmol_list[index]
    
    def get_index(self, mol: SMILES|Mol):
        smiles = convert2SMILES(mol)
        return self._smiles_to_index[smiles]
    
    @property
    def smiles_list(self):
        return self._smiles_list
    
    @property
    def rdmol_list(self):
        if self._rdmol_list is None:
            return [Chem.MolFromSmiles(smiles) for smiles in self._smiles_list]
        return self._rdmol_list
    
    @property
    def frequency_distribution(self):
        return self._frequency_distribution
    
    def load_library_file(self,
                          library_path: str,
                          use_frequency: True
                          ):
        extension = os.path.splitext(library_path)[1]
        assert extension in ['.csv', '.smi']

        with open(library_path) as f:
            frequency_list = None
            if extension == '.smi':
                smiles_list = [l.strip() for l in f.readlines()]
            else:
                header = f.readline().strip().split(',')
                if len(header) == 1:
                    smiles_list = [l.strip() for l in f.readlines()]
                else:
                    lines = [l.strip().split(',') for l in f.readlines()]
                    smiles_list = [smiles for smiles, _ in lines]
                    if use_frequency:
                        frequency_list = FloatTensor([float(f) for _, f in lines])
        return smiles_list, frequency_list
    
    @classmethod
    def create_from_library(cls,
                            library_path: str,
                            mol_list: list[Mol|SMILES],
                            save_frequency: bool = True,
                            cpus: int = 1,
                            ):
        from collections import Counter
        import parmap

        extension = os.path.splitext(library_path)[1]
        assert extension in ['.csv', '.smi']

        res = parmap.map(cls.decompose,
                         mol_list,
                         pm_processes=cpus,
                         pm_chunksize=1000,
                         pm_pbar=True
                         )
        
        block_list: list[SMILES] = []
        flag_list: list[bool] = []
        for blocks in res:
            if blocks is None:
                flag_list.append(False)
            else:
                flag_list.append(True)
                block_list.extend(blocks)

        block_freq_list = sorted(Counter(block_list).items(), key=lambda x: x[1], reverse=True)

        if save_frequency:
            assert extension == '.csv'
            with open(library_path, 'w') as f:
                f.write('SMILES,Frequency\n')
                for block, freq in block_freq_list:
                    block = convert2SMILES(block)
                    f.write(f'{block},{freq}\n')
        else:
            with open(library_path, 'w') as f:
                if extension == '.csv':
                    f.write('SMILES\n')
                for block, _ in block_freq_list:
                    block = convert2SMILES(block)
                    f.write(f'{block}\n')
        return flag_list
    
    @classmethod
    def decompose(cls,
                  mol: Mol|SMILES
                  ):
        try:
            res = cls.fragmentation.decompose()
            return res
        except:
            return None