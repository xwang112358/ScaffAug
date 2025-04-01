import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

# Activation functions
nn_act = torch.nn.ReLU()
F_act = F.relu

class GINConv(MessagePassing):
   def __init__(self, emb_dim):
       '''
       emb_dim (int): node embedding dimensionality
       '''
       super(GINConv, self).__init__(aggr="add")

       self.mlp = torch.nn.Sequential(
           torch.nn.Linear(emb_dim, 2*emb_dim), 
           torch.nn.BatchNorm1d(2*emb_dim), 
           nn_act, 
           torch.nn.Linear(2*emb_dim, emb_dim)
       )
       self.eps = torch.nn.Parameter(torch.Tensor([0]))
       
       # Add an edge embedding layer to transform edge features
       self.edge_embedding = torch.nn.Linear(4, emb_dim)  # Assuming edge_attr has 4 features

   def forward(self, x, edge_index, edge_attr):
       # Transform edge attributes if they exist
       if edge_attr is not None:
           edge_attr = self.edge_embedding(edge_attr)
       
       out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
       return out

   def message(self, x_j, edge_attr):
       if edge_attr is None:
           return F_act(x_j)
       return F_act(x_j + edge_attr)

   def update(self, aggr_out):
       return aggr_out


class GIN(torch.nn.Module):
   def __init__(self, 
                num_layer=3, 
                emb_dim=300, 
                drop_ratio=0.25, 
                graph_pooling="max"):
       '''
       Simple GIN model for a single task
       Input:
        num_layer (int): number of GNN layers
        emb_dim (int): node embedding dimensionality
        drop_ratio (float): dropout ratio
        graph_pooling (str): graph pooling type
       '''
       super(GIN, self).__init__()
       self.num_layer = num_layer
       self.drop_ratio = drop_ratio
       self.emb_dim = emb_dim
       
       if self.num_layer < 2:
           raise ValueError("Number of GNN layers must be greater than 1.")
       
       self.node_embedding = torch.nn.Linear(12, emb_dim)  # Assuming x has 12 features
       
       # Node embedding layers
       self.convs = torch.nn.ModuleList()
       self.batch_norms = torch.nn.ModuleList()
       
       for layer in range(num_layer):
           self.convs.append(GINConv(emb_dim))
           self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
       
       # Graph pooling function
       if graph_pooling == "sum":
           self.pool = global_add_pool
       elif graph_pooling == "mean":
           self.pool = global_mean_pool
       elif graph_pooling == "max":
           self.pool = global_max_pool
       else:
           raise ValueError("Invalid graph pooling type.")
       
       # Prediction layer for a single task
       self.predictor = torch.nn.Sequential(
           torch.nn.Linear(emb_dim, 2*emb_dim),
           torch.nn.BatchNorm1d(2*emb_dim),
           torch.nn.ReLU(),
           torch.nn.Dropout(drop_ratio),
           torch.nn.Linear(2*emb_dim, 1)
       )

   def forward(self, batched_data):
       x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
       
       # First embed the node features
       h = self.node_embedding(x)
       h_list = [h]
       
       # Process node features through GIN layers
       for layer in range(self.num_layer):
           h = self.convs[layer](h, edge_index, edge_attr)
           h = self.batch_norms[layer](h)
           
           if layer == self.num_layer - 1:
               # Remove activation for the last layer
               h = F.dropout(h, self.drop_ratio, training=self.training)
           else:
               h = F_act(h)
               h = F.dropout(h, self.drop_ratio, training=self.training)
           
           h_list.append(h)
       
       # Get final node representations
       node_representation = h_list[-1]
       
       # Apply graph pooling
       graph_representation = self.pool(node_representation, batch)
       
       # Predict output
       output = self.predictor(graph_representation)
       
       return output
   

if __name__ == "__main__":
    # Test the GIN model with batched graph data
    from torch_geometric.data import Data, Batch
    import numpy as np
    
    # Create two sample graphs
    # Graph 1: 3 nodes, 2 edges
    x1 = torch.randn(3, 300)  # 3 nodes with 300 features each
    edge_index1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()  # 2 edges
    edge_attr1 = torch.randn(2, 300)  # Edge features
    data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1)
    
    # Graph 2: 4 nodes, 3 edges
    x2 = torch.randn(4, 300)  # 4 nodes with 300 features each
    edge_index2 = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long).t()  # 3 edges
    edge_attr2 = torch.randn(3, 300)  # Edge features
    data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)
    
    # Create a batch from the two graphs
    batch = Batch.from_data_list([data1, data2])
    
    # Initialize the GIN model
    model = GIN(num_layer=3, emb_dim=300, drop_ratio=0.2, graph_pooling="max")
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(batch)
    
    # Print model information
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Print input information
    print("\nInput:")
    print(f"Number of graphs in batch: {batch.num_graphs}")
    print(f"Number of nodes: {batch.x.size(0)}")
    print(f"Number of edges: {batch.edge_index.size(1)}")
    print(f"Node feature dimensions: {batch.x.size(1)}")
    
    # Print output information
    print("\nOutput:")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.squeeze().tolist()}")
   

