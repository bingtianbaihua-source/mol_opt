import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from rdkit import Chem, RDLogger

import torch
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.data import Data as PyGData, Batch as PyGBatch

import os
import logging

from typing import Dict, Union, Optional, Set, Tuple, List
from rdkit.Chem import Mol
from torch import BoolTensor, FloatTensor
from src.utils import SMILES 

from src.fragmentation import BRICS_BlockLibrary
from src.transform import CoreGraphTransform
from src.transform import BlockGraphTransform
from src.model import BlockConnectionPredictor
from src.utils import convert2rdmol

from .generator_utils import *

RDLogger.DisableLog('rdApp.*')

TERMINATION = 1
ADDITION = 2
FAIL = 3

class MoleculeBuilder:
    def __init__(self, config) :
        self.model: BlockConnectionPredictor
        self.library: BRICS_BlockLibrary
        self.Z_library: FloatTensor
        self.library_mask: BoolTensor
        self.library_freq_weighted: FloatTensor
        self.window_size: int
        self.target_properties: Set[str]

        self.cfg = config
        self.max_iteration = config.max_iteration

        # Load Model & Library
        library_builtin_model_path = config.library_builtin_model_path
        if library_builtin_model_path is not None and os.path.exists(library_builtin_model_path) :
            self.model, self.library, self.Z_library, self.library_mask = \
                            self.load_library_builtin_model(library_builtin_model_path)
        else :
            self.model = model = self.load_model(config.model_path)
            self.library = library = self.load_library(config.library_path)
            self.Z_library = self.get_Z_library(model, library)
            self.library_mask = self.get_library_mask(library)
            if library_builtin_model_path is not None :
                self.save_builtin_model(library_builtin_model_path)

        # Setup Library Information
        print(Chem.MolToSmiles(self.library.get_rdmol(1)))
        library_freq = self.library.frequency_distribution
        self.library_freq_weighted = library_freq ** config.alpha
        self.window_size = min(len(self.library), config.window_size)

        # Setup after self.setup()
        self.target_properties = self.model.property_keys

    def generate(
        self,
        scaffold: Union[Mol, SMILES],
        condition: Optional[Dict[str, float]] = None,
        return_traj: bool = False,
        ) :
        if return_traj :
            return self.__generate_w_traj(scaffold, condition)
        else :
            return self.__generate_wo_traj(scaffold, condition)

    __call__ = generate
    run = generate

    def __generate_wo_traj(
        self,
        scaffold: Union[Mol, SMILES],
        condition: Optional[Dict[str, float]] = None,
        ) -> Mol :
        core_mol: Mol = convert2rdmol(scaffold)
        cond = self.model.standardize_property(condition)
        for _ in range(0, self.max_iteration) :
            step_sign, composed_mol = self.run_one_step(core_mol, standardized_condition=cond)
            if step_sign is TERMINATION :
                return core_mol
            elif step_sign is ADDITION :
                core_mol = composed_mol
            else : #step_sign is FAIL
                return None
        return None 

    def __generate_w_traj(
        self,
        scaffold: Union[Mol, SMILES],
        condition: Optional[Dict[str, float]] = None,
        ) -> Tuple[Mol, List[Mol]] :
        core_mol: Mol = convert2rdmol(scaffold)
        cond = self.model.standardize_property(condition)
        traj = [core_mol]
        for _ in range(0, self.max_iteration) :
            step_sign, composed_mol = self.run_one_step(core_mol, standardized_condition=cond)
            if step_sign is TERMINATION :
                return core_mol, traj
            elif step_sign is ADDITION :
                traj.append(composed_mol)
                core_mol = composed_mol
            else : #step_sign is FAIL
                traj.append(None)
                return None, traj
        return None, traj

    @torch.no_grad()
    def run_one_step(
        self,
        core_mol: Union[Mol, SMILES],
        condition: Dict[str, float] = None,
        standardized_condition: FloatTensor = None,
        ) -> Mol :

        core_mol: Mol = convert2rdmol(core_mol)

        assert (condition is not None) or (standardized_condition is not None)
        if standardized_condition is not None :
            cond = standardized_condition
        else :
            cond = self.model.standardize_property(condition)

        # Graph Embedding
        pygdata_core = CoreGraphTransform.call(core_mol)
        x_upd_core, Z_core = self.model.core_molecule_embedding(pygdata_core)
        
        # Condition Embedding
        x_upd_core, Z_core = self.model.condition_embedding(x_upd_core, Z_core, cond)

        # Predict Termination
        termination = self.predict_termination(Z_core)
        if termination :
            return TERMINATION, None

        # Sampling building blocks
        prob_dist_block = self.get_prob_dist_block(core_mol, Z_core)
                                                                                                # (N_lib)
        for _ in range(10) :
            if not torch.is_nonzero(prob_dist_block.sum()):
                return FAIL, None

            # Sample block
            block_idx = self.sample_block(prob_dist_block)
            block_mol = self.library.get_rdmol(block_idx)
            # print(Chem.MolToSmiles(block_mol))
            # print(Chem.MolToSmiles(self.library.get_rdmol(1)))

            # Predict Index
            Z_block = self.Z_library[block_idx].unsqueeze(0)
            atom_idx = self.predict_atom_idx(core_mol, block_mol, pygdata_core, x_upd_core, Z_core, Z_block)
            if atom_idx is None :
                prob_dist_block[block_idx] = 0
                continue

            # Compose
            composed_mol = compose(core_mol, block_mol, atom_idx, 0)
            if composed_mol is not None :
                return ADDITION, composed_mol
        return FAIL, None

    def predict_termination(self, Z_core) :
        p_term = self.model.get_termination_probability(Z_core)
        termination = Bernoulli(probs=p_term).sample().bool().item()
        return termination

    def get_prob_dist_block(self, core_mol, Z_core) :
        prob_dist_block = torch.zeros((len(self.library), )) 

        brics_labels = [int(label) for label in get_possible_brics_labels(core_mol)]
        block_index_list = torch.where(self.library_mask[brics_labels].sum(dim=0) > 0)[0]
        if block_index_list.size(0) == 0 :
            return prob_dist_block

        if self.window_size < block_index_list.size(0) :
            freq = self.library_freq_weighted[block_index_list]
            block_index_list = block_index_list[torch.multinomial(freq, self.window_size, False)]

        Z_core = Z_core.repeat(len(block_index_list), 1)
        Z_block = self.Z_library[block_index_list]
        priority_block = self.model.get_block_priority(Z_core, Z_block)
        prob_dist_block[block_index_list] = priority_block

        return prob_dist_block
    
    def sample_block(self, prob_dist_block) -> int:
        block_idx = Categorical(probs = prob_dist_block).sample().item()
        return block_idx

    def predict_atom_idx(self, core_mol, block_mol, pygdata_core, x_upd_core, Z_core, Z_block) -> Union[int, None]:
        edge_index_core, edge_attr_core = pygdata_core.edge_index, pygdata_core.edge_attr
        prob_dist_atom = self.model.get_atom_probability_distribution(
                x_upd_core, edge_index_core, edge_attr_core, Z_core, Z_block
        )
        # Masking
        masked_prob_dist_atom = torch.zeros_like(prob_dist_atom)
        print(Chem.MolToSmiles(block_mol))
        atom_idxs = [atom_idx for atom_idx, _ in get_possible_indexs(core_mol, block_mol)]
        masked_prob_dist_atom[atom_idxs] = prob_dist_atom[atom_idxs]

        # Sampling
        if not torch.is_nonzero(masked_prob_dist_atom.sum()) :
            return None
        else :
            # Choose Index
            atom_idx = Categorical(probs = masked_prob_dist_atom).sample().item()
            return atom_idx

    def load_model(self, model_path) :
        model = BlockConnectionPredictor.load_from_file(model_path, map_location = 'cpu')
        model.eval()
        return model

    def load_library(self, library_path) :
        return BRICS_BlockLibrary(library_path, save_rdmol = True)

    def load_library_builtin_model(self, library_builtin_model_path) :
        checkpoint = torch.load(library_builtin_model_path, map_location = 'cpu')
        model = BlockConnectionPredictor.load_from_checkpoint(checkpoint)
        model.eval()
        
        library = BRICS_BlockLibrary(smiles_list = checkpoint['library_smiles'], \
                                frequency_list = checkpoint['library_frequency'], \
                                save_rdmol = True)
        
        Z_library = checkpoint['Z_library']
        library_mask = checkpoint['library_mask'].T
        return model, library, Z_library, library_mask

    def get_library_mask(self, library) -> BoolTensor:
        library_mask = torch.zeros((len(library), 17), dtype=torch.bool)
        for i, brics_label in enumerate(library.brics_labels_list) : 
            allow_brics_label_list = BRICS_ENV_INT[brics_label]
            for allow_brics_label in allow_brics_label_list :
                library_mask[i, allow_brics_label] = True
        return library_mask.T   # (17, n_library)
    
    @torch.no_grad()
    def get_Z_library(self, model, library) -> FloatTensor:
        library_pygdata_list = [BlockGraphTransform.call(mol) for mol in self.library.rdmol_list]
        library_pygbatch = PyGBatch.from_data_list(library_pygdata_list)
        _, Z_library = self.model.building_block_embedding(library_pygbatch)
        return Z_library

    def save_builtin_model(self, save_path) :
        logging.info(f"Create Local File ({save_path})")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.model._cfg,
            'property_information': self.model.property_information,
            'library_smiles': self.library.smiles_list,
            'library_frequency': torch.clone(self.library.frequency_distribution),
            'Z_library': torch.clone(self.Z_library),
            'library_mask': torch.clone(self.library_mask.T),
        }, save_path)
