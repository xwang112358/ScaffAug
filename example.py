from welqrate.dataset import WelQrateDataset
from welqrate.models.gnn2d.GCN import GCN_Model 
import torch
from welqrate.train import train
import configparser
from torch.optim import AdamW



# AID1843_dataset_2d = WelQrateDataset(dataset_name = 'AID488997', root =f'../dataset_test', mol_repr ='2dmol')
# # AID1843_dataset_3d = WelQrateDataset(dataset_name = 'AID435034', root =f'../dataset_test', mol_repr ='3dmol')

# # print(AID1843_dataset_2d[0])
# # print(AID1843_dataset_3d[0])

# split_dict = AID1843_dataset_2d.get_idx_split(split_scheme ='random_cv2')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config = configparser.ConfigParser()
# config.read('./config/example.cfg')


# model = GCN_Model().to(device)

# train(model, AID1843_dataset_2d, config, device)
