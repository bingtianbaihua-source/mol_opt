import torch
from torch import nn, FloatTensor, LongTensor
from typing import OrderedDict, Tuple
from .layers import GraphEmbeddingModel
from .layers import ConditionEmbeddingModel
from .layers import PropertyPredictionModel
from .layers import TerminationPredictionModel
from .layers import BlockSelectionModel
from .layers import AtomSelectionModel
from transform import NUM_ATOM_FEATURES, NUM_BOND_FEATURES
from transform import NUM_BLOCK_FEATURES
from torch_geometric.data import Data, Batch
from torch_geometric.typing import Adj
from src.utils import PropertyVector, GraphVector, NodeVector, EdgeVector

class BlockConnectionPredictor(nn.Module):

    def __init__(self,
                 cfg,
                 property_information: OrderedDict[str, Tuple[float, float]]
                 ) -> None:
        super(BlockConnectionPredictor, self).__init__()
        self._cfg = cfg
        self.property_information = property_information
        self.property_keys = property_information.keys()
        self.property_dim = len(property_information)

        self.core_graph_embedding_model = GraphEmbeddingModel(
            NUM_ATOM_FEATURES,
            NUM_BOND_FEATURES,
            0,
            **cfg.GraphEmbeddingModel_Core
        )
        self.block_graph_embedding_model = GraphEmbeddingModel(
            NUM_ATOM_FEATURES,
            NUM_BOND_FEATURES,
            NUM_BLOCK_FEATURES,
            **cfg.GraphEmbeddingModel_Block
        )

        self.property_prediction_model = PropertyPredictionModel(
            property_dim = self.property_dim,
            **cfg.PropertyPredictionModel
        )

        self.condition_embedding_model = ConditionEmbeddingModel(
            condition_dim = self.property_dim,
            **cfg.ConditionEmbeddingModel
        )

        self.termination_prediction_model = TerminationPredictionModel(
            **cfg.TerminationPredictionModel
        )

        self.block_selection_model = BlockSelectionModel(
            **cfg.BlockSelectionModel
        )

        self.atom_selection_model = AtomSelectionModel(
            NUM_BOND_FEATURES,
            **cfg.AtomSelectionModel
        )

    def standardize_property(
            self,
            property: dict[str, float|FloatTensor]
    ):
        assert property.keys() == self.property_keys
        property = [(property[key] - mean) / std
            for key, (mean,std) in self.property_information.items()]

        if isinstance(property[0], float):
            property = FloatTensor([property])
        else:
            property = torch.stack(property, dim=-1)

        return property

    def core_molecule_embedding(
            self,
            batch: Data|Batch
    ):
        return self.core_graph_embedding_model.forward_batch(batch)
    
    def building_block_embedding(
            self,
            batch: Data|Batch
    ):
        return self.block_graph_embedding_model.forward_batch(batch)
    
    def get_property_prediction(
            self,
            Z_core: GraphVector
    ):
        return self.property_prediction_model(Z_core)
    
    def condition_embedding(
            self,
            x_upd_core: NodeVector,
            Z_core: GraphVector,
            condition: PropertyVector,
            node2graph_core: LongTensor|None
    ):
        return self.condition_embedding_model(
            x_upd_core,
            Z_core,
            condition,
            node2graph_core
        )
    
    def get_termination_logit(self,
                              Z_core: GraphVector,
                              ):
        return self.termination_prediction_model(Z_core, return_logit = True)
    
    def get_termination_probability(self,
                                    Z_core: GraphVector
                                    ):
        return self.termination_prediction_model(Z_core)
    
    def get_blocck_priority(self,
                            Z_core: GraphVector,
                            Z_block):
        return self.block_selection_model(Z_core,Z_block)
    
    def get_atom_probability_distribution(
            self,
            x_upd_core: NodeVector,
            edge_index_core: Adj,
            edge_attr_core: EdgeVector,
            Z_core: GraphVector,
            Z_block: GraphVector,
            node2graph_core: LongTensor|None
    ):
        return self.atom_selection_model(
            x_upd_core,
            edge_index_core,
            edge_attr_core,
            Z_core,
            Z_block,
            node2graph_core
        )
    
    def get_atom_logit(
            self,
            x_upd_core: NodeVector,
            edge_index_core: Adj,
            edge_attr_core: EdgeVector,
            Z_core: GraphVector,
            Z_block: GraphVector,
            node2graph_core: LongTensor|None
    ):
        return self.atom_selection_model(
            x_upd_core,
            edge_index_core,
            edge_attr_core,
            Z_core,
            Z_block,
            node2graph_core,
            return_logit = True
        )
    
    def initialize_parameter(self):
        for param in self.parameters():
            if param.dim() == 1:
                continue
            else:
                nn.init.xavier_normal_(param)

    def save(self, save_path):
        torch.save({'model_state_dict': self.state_dict(),
                    'config': self._cfg,
                    'property_information': self.property_information
        }, save_path)

    @classmethod
    def load_from_file(
        cls, 
        checkpoint_path, 
        map_location='cpu'
        ):
        checkpoint = torch.load(checkpoint_path, 
                                map_location = map_location)
        return cls.load_from_checkpoint(
            checkpoint,
            map_location = map_location
        )
    
    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint,
            map_location = 'cpu'
    ):
        model = cls(checkpoint['config'],
                    checkpoint['property_information'])
        model.load_state_dict(
            checkpoint['model_state_dict']
        )
        model.to(map_location)
        return model