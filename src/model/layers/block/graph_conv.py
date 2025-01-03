from torch_geometric.nn.conv import MessagePassing
from typing import Union, Tuple, Optional

class GraphConv(MessagePassing):
    def __init__(self,
                 node_dim: Union[int, Tuple[int]],
                 edge_dim: int,
                 activation: Optional[str] = None,
                 norm: Optional[str] = None,
                 dropout: float = 0.0,
                 **kwargs,
                 ):
        super(GraphConv, self).__init__(**kwargs)

        if isinstance(node_dim, int):
            src_node_dim, dst_node_dim = node_dim, node_dim
        else:
            src_node_dim, dst_node_dim = node_dim

        