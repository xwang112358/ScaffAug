from welqrate.dataset import WelQrateDataset
from welqrate.models.gnn2d.GCN import GCN_Model 
import torch
from welqrate.train import train
import yaml
import itertools
import copy
from pathlib import Path
import pandas as pd
import csv
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AID1798', required=True)
parser.add_argument('--split', type=str, default='random_cv1', required=True)
args = parser.parse_args()
# Define hyperparameter search space
hyperparams = {
    'hidden_channels': [32, 64, 128], # 64, 128
    'num_layers': [2, 3, 4], #  3, 4
    'peak_lr': [1e-2, 1e-3, 1e-4]
}

# Load base config
with open('./configs/gcn.yaml') as file:
    base_config = yaml.safe_load(file)

# Setup dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = WelQrateDataset(dataset_name=args.dataset, root='./welqrate_datasets', mol_repr='2dmol')

# Create results directory
Path('results').mkdir(exist_ok=True)

# Initialize results DataFrame
results_data = []

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(
    hyperparams['hidden_channels'],
    hyperparams['num_layers'],
    hyperparams['peak_lr']
))

# Get dataset name and split scheme from config
dataset_name = args.dataset
split_scheme = args.split

# Create CSV file with headers
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f'results/gcn_finetuning_{dataset_name}_{split_scheme}_{timestamp}.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'hidden_channels', 'num_layers', 'peak_lr',
        'test_logAUC', 'test_EF', 'test_DCG', 'test_BEDROC'
    ])

# Also update the best parameters filename
best_params_file = f'results/best_parameters_{dataset_name}_{split_scheme}_{timestamp}.txt'

# Run training for each combination
for hidden_ch, n_layers, lr in param_combinations:
    try:
        # Update config
        config = copy.deepcopy(base_config)
        config['MODEL']['hidden_channels'] = hidden_ch
        config['MODEL']['num_layers'] = n_layers
        config['TRAIN']['peak_lr'] = lr
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
        test_logAUC, test_EF, test_DCG, test_BEDROC = train(model, dataset, config, device)
        
        # Extract metrics
        test_metrics = [
            test_logAUC,
            test_EF,
            test_DCG,
            test_BEDROC
        ]
        
        # Save results to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hidden_ch, n_layers, lr] + test_metrics)
        
        # Store results for DataFrame
        results_data.append({
            'hidden_channels': hidden_ch,
            'num_layers': n_layers,
            'peak_lr': lr,
            'test_logAUC': test_logAUC,
            'test_EF': test_EF,
            'test_DCG': test_DCG,
            'test_BEDROC': test_BEDROC
        })
        
    except Exception as e:
        print(f"Error occurred with parameters: hidden_ch={hidden_ch}, n_layers={n_layers}, lr={lr}")
        print(f"Error message: {str(e)}")
        
        # Save error in CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hidden_ch, n_layers, lr, 'ERROR', 'ERROR', 'ERROR', 'ERROR'])

# # Create DataFrame and save as CSV
# df = pd.DataFrame(results_data)

# # Find best parameters for each metric
# best_params = {
#     'logAUC': df.loc[df['test_logAUC'].idxmax()],
#     'EF': df.loc[df['test_EF'].idxmax()],
#     'DCG': df.loc[df['test_DCG'].idxmax()],
#     'BEDROC': df.loc[df['test_BEDROC'].idxmax()]
# }

# # Save best parameters to a separate file
# with open(best_params_file, 'w') as f:
#     f.write(f"Best parameters for {dataset_name} dataset with {split_scheme} split:\n\n")
#     for metric, params in best_params.items():
#         f.write(f"\nBest for {metric}:\n")
#         for key, value in params.items():
#             f.write(f"{key}: {value}\n")

print("\nFinetuning completed. Results saved in:", csv_file)
