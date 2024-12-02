import torch
from welqrate.dataset import WelQrateDataset
from welqrate.models.gnn2d.GCN import GCN_Model 
from welqrate.train_aug import train as train_aug
from welqrate.train import train as train
from welqrate.train_pseudo_label import train as train_pseudo_label

import yaml
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gcn_model = GCN_Model(in_channels=12, hidden_channels=64, num_layers=4).to(device)

with open('./configs/gcn.yaml') as file:
    config = yaml.safe_load(file)

config['TRAIN']['peak_lr'] = 1e-3
# load the augmented dataset 
augmented_dataset = torch.load('./welqrate_datasets/augmentation/AID1798_random_cv1_0.1_generated_graphs.pt')
# load the original dataset
original_dataset = WelQrateDataset(dataset_name='AID1798', root='./welqrate_datasets', mol_repr='2dmol')

# test_logAUC, test_EF, test_DCG, test_BEDROC = train_aug(gcn_model, original_dataset, 
#                                                     augmented_dataset, config, device, aug_weight=0.1)

# test_logAUC, test_EF, test_DCG, test_BEDROC = train(gcn_model, original_dataset, config, device)


test_logAUC, test_EF, test_DCG, test_BEDROC = train_pseudo_label(gcn_model, original_dataset, 
                                                    augmented_dataset, config, device, aug_weight=0.1)