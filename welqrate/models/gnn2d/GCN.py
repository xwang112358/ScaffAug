from .GNNConv import GCN as Encoder
from torch_geometric.nn import global_add_pool
import torch
from torch.nn import Linear, ReLU, Dropout

class GCN_Model(torch.nn.Module):

    def __init__(
            self,
            in_channels = 28,
            hidden_channels = 32,
            num_layers = 3,
            dropout=0.0,
            act='relu',
            one_hot=False,
            act_first=False,
            act_kwargs=None,
            norm=None,
            norm_kwargs=None,
            jk=None
    ):
        super(GCN_Model, self).__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            jk=jk
        )
        self.pool = global_add_pool
        self.one_hot = one_hot
        # self.ffn_dropout = Dropout(p=0.25)
        self.lin1 = Linear(hidden_channels, 64)
        self.lin2 = Linear(64, 1)
        self.activate_func = ReLU()
        self.ff_dropout = Dropout(p=0.25)

    def forward(self, batch_data):
        if self.one_hot:
            x = batch_data.one_hot_atom
            # convert one hot to float
            x = x.float()
            edge_index = batch_data.edge_index
            node_embedding = self.encoder(x, edge_index)
        else:   
            x = batch_data.x
            edge_index = batch_data.edge_index
            node_embedding = self.encoder(x, edge_index)
        graph_embedding = self.pool(node_embedding, batch_data.batch)
        
        # Apply dropout after first linear layer and activation
        x = self.lin1(graph_embedding)
        x = self.activate_func(x)
        x = self.ff_dropout(x)
        
        # Final prediction without dropout
        prediction = self.lin2(x)

        return prediction
