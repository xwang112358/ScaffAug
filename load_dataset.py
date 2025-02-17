from welqrate.dataset import WelQrateDataset
from torch_geometric.data import Data
import time
import torch
datasets = ['AID1798', 'AID463087', 'AID488997', 'AID2689', 'AID485290']


# for dataset_name in datasets:
#     print(f"\nProcessing {dataset_name}")
#     dataset = WelQrateDataset(dataset_name=dataset_name, root='./welqrate_datasets', mol_repr='2dmol')
#     print(f"Dataset size: {len(dataset)}")
#     print(f"Sample data point: {dataset[0]}")
    
#     # only keep x, edge_index, edge_attr, y for each data point and save it in a list
#     start_time = time.time()
#     data_list = []
#     for graph in dataset:
#         data_list.append(Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=graph.y))

#     torch.save(data_list, f'./welqrate_datasets/simple/{dataset_name}_data_list.pt')
#     print(f"Processed data list size: {len(data_list)}")
#     print(f"Sample processed data point: {data_list[0]}")
#     end_time = time.time()
#     print(f"Time taken: {end_time - start_time} seconds")


augmented_dataset = torch.load(f'./augment_pyg_datasets/AID1798_random_cv1_0.1_generated_pyg_graphs.pt')
print(type(augmented_dataset))






