from welqrate.dataset import WelQrateDataset
from torch_geometric.data import Data
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
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

orig_dataset = WelQrateDataset(dataset_name='AID1798', root='./welqrate_datasets', mol_repr='2dmol')
aug_dataset = torch.load(f'./augment_pyg_graphs_labels/AID1798_random_cv1_0.1_augment_pyg_graphs_labels.pt')

print(aug_dataset[0].y)
# store data objects from orig_dataset in a list
orig_data_list = []
for graph in tqdm(orig_dataset, desc="Processing original dataset"):
    orig_data_list.append(Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=graph.y))


print(orig_data_list[0].y)

dataset_list = orig_data_list + aug_dataset

# check if y is Nonetype
num_train_active = len(torch.nonzero(torch.tensor([data.y for data in dataset_list])))
print(f"Number of active molecules: {num_train_active}")

# # call dataloader
# dataloader = DataLoader(dataset_list, batch_size=128, shuffle=True)

# print(orig_data_list[0])









