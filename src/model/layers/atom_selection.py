from torch import nn
from .graph_embedding import GraphEmbeddingModel
from .block.fc import Linear
from torch import FloatTensor, LongTensor
from torch_geometric.typing import Adj
import torch
from torch_scatter.composite import scatter_softmax

class AtomSelectionModel(nn.Module):

    def __init__(self, 
                 core_edge_input_dim: int,
                 core_node_vector_dim: int,
                 core_graph_vector_dim: int,
                 block_graph_vector_dim: int,
                 hidden_dim: int = 128,
                 n_block: int = 2,
                 dropout: float = 0.0
                 ) -> None:
        super(AtomSelectionModel).__init__()

        self.graph_embedding = GraphEmbeddingModel(
            node_input_dim=hidden_dim,
            edge_input_dim=core_edge_input_dim,
            global_input_dim=core_graph_vector_dim + block_graph_vector_dim,
            hidden_dim=hidden_dim,
            graph_vector_dim=None,
            n_block=n_block,
            dropout=dropout
        )

        self.mlp = nn.Sequential(
            Linear(
                input_dim=core_node_vector_dim,
                output_dim=hidden_dim,
                activation='relu',
                dropout=dropout
            ),
            Linear(
                input_dim=hidden_dim,
                output_dim=1,
                activation=None
            )
        )

    def forward(
            self,
            x_upd_core: FloatTensor,
            edge_index_core: Adj,
            edge_attr_core: FloatTensor,
            Z_core: FloatTensor,
            Z_block: FloatTensor,
            node2graph_core: Optional[LongTensor] = None,
            return_logit: bool=False
    ):
        Z_cat = torch.cat([Z_core, Z_block], dim=-1)
        x_upd2, _ = self.graph_embedding(
            x_upd_core,
            edge_index_core,
            edge_attr_core,
            global_x = Z_cat,
            node2graph = node2graph_core
        )

        logit = self.mlp(x_upd2).squeeze(-1)

        if return_logit:
            return logit
        else:
            if node2graph_core is not None:
                p = scatter_softmax(logit, node2graph_core)
            else:
                p = torch.softmax(logit, dim=-1)
            return p
