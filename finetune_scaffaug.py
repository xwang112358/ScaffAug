import torch
from welqrate.dataset import WelQrateDataset
from welqrate.models.gnn2d.GCN import GCN_Model 
from welqrate.train_aug import train as train_aug
from welqrate.train import train as train
from welqrate.train_pseudo_label import train as train_pseudo_label
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AID1798', required=True)
parser.add_argument('--split', type=str, default='random_cv1', required=True)
parser.add_argument('--ratio', type=float, default=0.1, required=True)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gcn_model = GCN_Model(in_channels=12, hidden_channels=64, num_layers=4).to(device)

with open('./configs/gcn.yaml') as file:
    config = yaml.safe_load(file)

config['TRAIN']['peak_lr'] = 1e-3
# load the augmented dataset 
augmented_dataset = torch.load(f'./welqrate_datasets/augmentation/{args.dataset}_{args.split}_{args.ratio}_generated_graphs.pt')
# load the original dataset
original_dataset = WelQrateDataset(dataset_name=args.dataset, root='./welqrate_datasets', mol_repr='2dmol')



test_logAUC, test_EF, test_DCG, test_BEDROC = train_pseudo_label(gcn_model, original_dataset, 
                                                    augmented_dataset, config, device, aug_weight=0.1)