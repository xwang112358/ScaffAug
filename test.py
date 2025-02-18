import torch
from welqrate.dataset import WelQrateDataset
from welqrate.models.gnn2d.GCN import GCN_Model 
# from welqrate.self_train import train as train_pseudo_label
from welqrate.train_pseudo_label import train as train_pseudo_label
from welqrate.train_aug import train as train_aug
from welqrate.loader import get_train_loader, get_self_train_loader
import argparse
import yaml
import itertools
import copy
import os
import csv
from datetime import datetime
import pandas as pd

# ----------------- Config -----------------
dataset_name = 'AID1798'
split_scheme = 'random_cv1'

with open('./configs/scaffaug.yaml') as file:
    base_config = yaml.safe_load(file)

base_config['TRAIN']['num_epochs'] = 100
base_config['GENERAL']['seed'] = 1
base_config['AUGMENTATION']['confidence_threshold'] = 0.8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------- Load Data -----------------
original_dataset = WelQrateDataset(dataset_name=dataset_name, root='./welqrate_datasets', mol_repr='2dmol')
augmented_dataset = torch.load(f'./augment_pyg_graphs_labels/{dataset_name}_{split_scheme}_0.1_augment_pyg_graphs_labels.pt')



model = GCN_Model(
    in_channels=12,
    hidden_channels=128,
    num_layers=3,
).to(device)


train_aug(
    model, original_dataset, augmented_dataset, base_config, device
)

# ----------------- Test CombinedDataset -----------------
# from welqrate.self_train import AugmentedDataset, CombinedDataset

# # Create AugmentedDataset from augmented_dataset
# aug_dataset = AugmentedDataset(augmented_dataset)
# print(f'Augmented dataset: {aug_dataset[0]}')

# split_dict = original_dataset.get_idx_split(split_scheme)
# train_data = original_dataset[split_dict['train']]

# print(f'train_data length: {len(train_data)}')
# print(f'aug_dataset length: {len(aug_dataset)}')
# combined_dataset = CombinedDataset(train_data, aug_dataset)
# print(f'Combined dataset length: {len(combined_dataset)}')



# train_loader = get_self_train_loader(combined_dataset, 
#                                 batch_size=128, 
#                                 num_workers=0, 
#                                 seed=1)

# batches = list(train_loader)




