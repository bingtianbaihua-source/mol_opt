# import torch
# import torch.nn as nn
# from typing import Optional, Tuple, Union
# from torch import FloatTensor, LongTensor
# from torch_geometric.data import Data as PyGData, Batch as PyGBatch
# from torch_geometric.typing import Adj
# from e3nn import o3
# from e3nn.nn import FullyConnectedNet, Gate
# from e3nn.o3 import Irreps, spherical_harmonics
# from bbar.utils.typing import NodeVector, EdgeVector, GlobalVector, GraphVector

# class GraphEmbeddingModel(nn.Module):
#     def __init__(
#         self,
#         node_input_dim: int, 
#         edge_input_dim: int,
#         global_input_dim: Optional[int] = 0, 
#         hidden_dim: int = 128,
#         graph_vector_dim: Optional[int] = 0, 
#         n_block: int = 2, 
#         dropout: float = 0.0,
#     ):
#         super(GraphEmbeddingModel, self).__init__()

#         global_input_dim = global_input_dim or 0
#         graph_vector_dim = graph_vector_dim or 0

#         irreps_node = Irreps(f"{node_input_dim + global_input_dim}x0e")
#         irreps_edge = Irreps(f"{edge_input_dim}x0e")
#         irreps_hidden = Irreps(f"{hidden_dim}x0e + {hidden_dim//2}x1o")
#         print(f"irreps_node_dim: {irreps_node.dim} {irreps_edge.dim} {irreps_hidden.dim}")

#         # Modified: Update node and edge embedding layers to use FullyConnectedNet
#         self.node_embedding = FullyConnectedNet([irreps_node.dim, irreps_hidden.dim], torch.nn.functional.silu)
#         self.edge_embedding = FullyConnectedNet([irreps_edge.dim, irreps_hidden.dim - 3], torch.nn.functional.silu)

#         # Modified: Use EquivariantResidualBlock
#         self.blocks = nn.ModuleList([
#             EquivariantResidualBlock(
#                 irreps_node=irreps_hidden, irreps_edge=irreps_hidden, dropout=dropout)
#             # print("==="),
#             for _ in range(n_block)
#         ])

#         self.final_node_embedding = FullyConnectedNet([hidden_dim + irreps_node.dim, hidden_dim], torch.nn.functional.silu)

#         if graph_vector_dim > 0:
#             self.readout = EquivariantReadout(
#                 irreps_node=irreps_hidden,
#                 hidden_dim=graph_vector_dim,
#                 output_dim=graph_vector_dim,
#                 global_input_dim=global_input_dim,
#                 dropout=dropout
#             )
#         else:
#             self.readout = None

#     def forward(
#         self,
#         x_inp: NodeVector,
#         edge_index: Adj,
#         edge_attr: EdgeVector,
#         global_x: Optional[GlobalVector] = None,
#         node2graph: Optional[LongTensor] = None,
#         pos = None
#     ) -> Tuple[NodeVector, Optional[GraphVector]]:
#         x = self.concat(x_inp, global_x, node2graph)

#         x_emb = self.node_embedding(x)
#         edge_attr = self.edge_embedding(edge_attr)
#         # print(f"x_emb: {x_emb.shape[-1]}")
#         # print(f"x_emb: {self.blocks[0].conv._in1_dim}")

#         for convblock in self.blocks:
#             x_emb = convblock(x_emb, edge_index, edge_attr, node2graph, pos)

#         x_emb = torch.cat([x_emb, x_inp], dim=-1)
#         x_emb = self.final_node_embedding(x_emb)

#         if self.readout is not None:
#             Z = self.readout(x_emb, node2graph, global_x)
#         else:
#             Z = None

#         return x_emb, Z

#     def forward_batch(self, batch: Union[PyGBatch, PyGData]) -> Tuple[NodeVector, Optional[GraphVector]]:
#         if isinstance(batch, PyGBatch):
#             node2graph = batch.batch
#         else:
#             node2graph = None

#         global_x = batch.get('global_x', None)

#         return self.forward(batch.x, batch.edge_index, batch.edge_attr, global_x, node2graph, batch.pos)

#     def concat(self, x: NodeVector, global_x: Optional[GlobalVector], node2graph: LongTensor) -> FloatTensor:
#         if global_x is not None:
#             if node2graph is None:
#                 global_x = global_x.repeat(x.size(0), 1)
#             else:
#                 global_x = global_x[node2graph]
#             x = torch.cat([x, global_x], dim=-1)
#         return x

# class EquivariantResidualBlock(nn.Module):
#     def __init__(self, irreps_node, irreps_edge, dropout=0.0):
#         super().__init__()
#         self.irreps_node = irreps_node
#         self.conv = o3.FullyConnectedTensorProduct(irreps_node, irreps_edge, irreps_node)
#         # print(self.conv._in2_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, edge_index, edge_attr, node2graph, pos):
#         row, col = edge_index
#         edge_vec = pos[row] - pos[col]
#         edge_sh = spherical_harmonics(l=1, x=edge_vec, normalize=True, normalization='component')
#         # print(f"edge_sh dim:{edge_sh.shape[-1]}")
        
#         # Ensure edge_attr and edge_sh have compatible dimensions
#         edge_attr = torch.cat((edge_attr, edge_sh), dim=-1)
#         print(f"x dim: {x.shape}")
#         print(f"y dim: {edge_attr.shape}")

#         assert edge_attr.shape[-1] == self.conv.irreps_in2.dim, \
#             f"edge_attr last dimension must be {self.conv.irreps_in2.dim}, got {edge_attr.shape[-1]}"

#         x_out = self.conv(x, edge_attr)
#         return x + self.dropout(x_out)

# class EquivariantReadout(nn.Module):
#     def __init__(self, irreps_node, hidden_dim, output_dim, global_input_dim=0, dropout=0.0):
#         super().__init__()
#         self.irreps_node = irreps_node
#         self.fc = FullyConnectedNet([irreps_node.dim, hidden_dim, output_dim], torch.nn.functional.silu)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, node2graph, global_x):
#         x = self.fc(x)
#         x = self.dropout(x)
#         if global_x is not None:
#             x = torch.cat([x, global_x[node2graph]], dim=-1)
#         return x

import torch
from torch_geometric.data import Data
from e3nn import o3
from e3nn.o3 import Irreps
from e3nn.nn import Gate
from e3nn.nn.models.v2103.gate_points_networks import MessagePassing

class NetworkForAGraphWithAttributes(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        max_radius,
        num_neighbors,
        num_nodes,
        mul=50,
        layers=3,
        lmax=2,
        pool_nodes=True,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = 10
        self.num_nodes = num_nodes
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.pool_nodes = pool_nodes

        irreps_node_hidden = o3.Irreps([(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_node_input] + layers * [irreps_node_hidden] + [irreps_node_output],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr + o3.Irreps.spherical_harmonics(lmax),
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
        )

        self.irreps_node_input = self.mp.irreps_node_input
        self.irreps_node_attr = self.mp.irreps_node_attr
        self.irreps_node_output = self.mp.irreps_node_output

    def preprocess(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        # Create graph
        if "edge_index" in data:
            edge_src = data["edge_index"][0]
            edge_dst = data["edge_index"][1]
        else:
            edge_index = radius_graph(data["pos"], self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]

        # Edge attributes
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]

        if "x" in data:
            node_input = data["x"]
        else:
            node_input = data["node_input"]

        node_attr = data["node_attr"]
        edge_attr = data["edge_attr"]

        return batch, node_input, node_attr, edge_attr, edge_src, edge_dst, edge_vec

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch, node_input, node_attr, edge_attr, edge_src, edge_dst, edge_vec = self.preprocess(data)
        del data

        # Edge attributes
        edge_sh = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization="component")
        edge_attr = torch.cat([edge_attr, edge_sh], dim=1)

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="smooth_finite",  # the smooth_finite basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        node_outputs = self.mp(node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

        if self.pool_nodes:
            return scatter(node_outputs, batch, int(batch.max()) + 1).div(self.num_nodes**0.5)
        else:
            return node_outputs