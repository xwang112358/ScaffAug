import torch
from welqrate.dataset import WelQrateDataset
from welqrate.models.gnn2d.GCN import GCN_Model 
from welqrate.train_pseudo_label import train as train_pseudo_label
import argparse
import yaml
import itertools
import copy
from pathlib import Path
import csv
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AID1798', required=True)
parser.add_argument('--split', type=str, default='random_cv1', required=True)
parser.add_argument('--ratio', type=float, default=0.1, required=True)
args = parser.parse_args()

# Define hyperparameter search space
# hyperparams = {
#     'hidden_channels': [32, 64, 128],
#     'num_layers': [2, 3, 4],
#     'peak_lr': [1e-2, 1e-3, 1e-4],
#     'aug_weight': [0.2],
#     'confidence_threshold': [0.6]
# }

hyperparams = {
    'hidden_channels': [128],
    'num_layers': [3],
    'peak_lr': [1e-3],
    'aug_weight': [0.2],
    'confidence_threshold': [0.6]
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load base config
with open('./configs/scaffaug.yaml') as file:
    base_config = yaml.safe_load(file)

# Create results directory and CSV file
Path('results').mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f'results/scaffaug_gcn_finetuning_{args.dataset}_{args.split}_{args.ratio}_{timestamp}.csv'

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'hidden_channels', 'num_layers', 'peak_lr', 'aug_weight', 'confidence_threshold',
        'test_logAUC', 'test_EF', 'test_DCG', 'test_BEDROC'
    ])

# Load datasets
augmented_dataset = torch.load(f'./augment_pyg_datasets/{args.dataset}_{args.split}_{args.ratio}_generated_pyg_graphs.pt')
original_dataset = WelQrateDataset(dataset_name=args.dataset, root='./welqrate_datasets', mol_repr='2dmol')

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(
    hyperparams['hidden_channels'],
    hyperparams['num_layers'],
    hyperparams['peak_lr'],
    hyperparams['aug_weight'],
    hyperparams['confidence_threshold']
))

# Run training for each combination
for hidden_ch, n_layers, lr, aug_weight, confidence_threshold in param_combinations:
    try:
        # Update config
        config = copy.deepcopy(base_config)
        config['MODEL']['hidden_channels'] = hidden_ch
        config['MODEL']['num_layers'] = n_layers
        config['TRAIN']['peak_lr'] = lr
        config['AUGMENTATION']['aug_weight'] = aug_weight
        config['AUGMENTATION']['confidence_threshold'] = confidence_threshold
        config['DATA']['split_scheme'] = args.split

        # Initialize model with current params
        model = GCN_Model(
            in_channels=12,
            hidden_channels=hidden_ch,
            num_layers=n_layers,
        ).to(device)
        
        print(f"\nTraining with parameters:")
        print(f"Hidden channels: {hidden_ch}")
        print(f"Number of layers: {n_layers}")
        print(f"Peak learning rate: {lr}")
        
        # Train model and get metrics
        test_logAUC, test_EF, test_DCG, test_BEDROC = train_pseudo_label(
            model, original_dataset, augmented_dataset, config, device
        )
        
        # Save results to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hidden_ch, n_layers, lr, aug_weight, confidence_threshold, test_logAUC, test_EF, test_DCG, test_BEDROC])
            
    except Exception as e:
        print(f"Error occurred with parameters: hidden_ch={hidden_ch}, n_layers={n_layers}, lr={lr}")
        print(f"Error message: {str(e)}")
        
        # Save error in CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hidden_ch, n_layers, lr, aug_weight, confidence_threshold, 'ERROR', 'ERROR', 'ERROR', 'ERROR'])

print("\nFinetuning completed. Results saved in:", csv_file)