from welqrate.dataset import WelQrateDataset
from welqrate.models.gnn2d.GCN import GCN_Model 
import torch
from welqrate.train import train as train
from welqrate.train_aug import train as train_aug
import yaml
import itertools
import copy
import os
import pandas as pd
import csv
from datetime import datetime
import argparse
import optuna

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AID1798', required=True)
parser.add_argument('--split', type=str, default='random_cv1', required=True)
parser.add_argument('--aug', action='store_true')
parser.add_argument('--valid',action='store_true')
args = parser.parse_args()

# Load base config
with open('./configs/gcn.yaml') as file:
    base_config = yaml.safe_load(file)

# Setup dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = WelQrateDataset(dataset_name=args.dataset, root='./welqrate_datasets', mol_repr='2dmol')

# Get dataset name and split scheme from config
dataset_name = args.dataset
split_scheme = args.split

# Create CSV file with headers
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.aug:
    results_dir = 'results_aug'
    csv_file = f'results_aug/gcn_finetuning_{dataset_name}_{split_scheme}_{timestamp}.csv'
elif args.valid:
    results_dir = 'results_valid_aug'
    csv_file = f'results_valid_aug/gcn_finetuning_{dataset_name}_{split_scheme}_{timestamp}.csv'
else:
    results_dir = 'results'
    csv_file = f'results/gcn_finetuning_{dataset_name}_{split_scheme}_{timestamp}.csv'

os.makedirs(results_dir, exist_ok=True)

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'hidden_channels', 'num_layers', 'peak_lr',
        'test_logAUC', 'test_EF', 'test_DCG', 'test_BEDROC',
        'test_EF500', 'test_EF1000', 'test_DCG500', 'test_DCG1000'
    ])

def objective(trial):
    # Define hyperparameter search space
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    peak_lr = trial.suggest_float('peak_lr', 1e-4, 1e-2, log=True)
    
    try:
        # Update config
        config = copy.deepcopy(base_config)
        config['MODEL']['hidden_channels'] = hidden_channels
        config['MODEL']['num_layers'] = num_layers
        config['TRAIN']['peak_lr'] = peak_lr
        config['DATA']['split_scheme'] = args.split
        
        # Initialize model with current params
        model = GCN_Model(
            in_channels=12,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        ).to(device)
        
        print(f"\nTrial {trial.number}")
        print(f"Hidden channels: {hidden_channels}")
        print(f"Number of layers: {num_layers}")
        print(f"Peak learning rate: {peak_lr}")
        
        # Train model and get metrics
        if args.aug:
            aug_dataset = torch.load(f'./augment_pyg_graphs_labels/{args.dataset}_{args.split}_0.1_augment_pyg_graphs_labels.pt')
            test_logAUC, test_EF100, test_DCG100, test_BEDROC, _, _, _, _ = train_aug(model=model, 
                                                                                      dataset=dataset, 
                                                                                      aug_dataset=aug_dataset, 
                                                                                      config=config, 
                                                                                      device=device, 
                                                                                      results_dir=results_dir)
        elif args.valid:
            aug_dataset = torch.load(f'./augment_valid_pyg_graphs_labels/{args.dataset}_{args.split}_0.1_augment_valid_pyg_graphs_labels.pt')
            test_logAUC, test_EF100, test_DCG100, test_BEDROC, _, _, _, _ = train_aug(model=model, 
                                                                                      dataset=dataset, 
                                                                                      aug_dataset=aug_dataset, 
                                                                                      config=config, 
                                                                                      device=device, 
                                                                                      results_dir=results_dir)
        else:
            test_logAUC, test_EF100, test_DCG100, test_BEDROC, _, _, _, _ = train(model=model, 
                                                                                  dataset=dataset, 
                                                                                  config=config, 
                                                                                  device=device)
        
        # Save results to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hidden_channels, num_layers, peak_lr, 
                             test_logAUC, test_EF100, test_DCG100, test_BEDROC])
        
        return test_BEDROC
        
    except Exception as e:
        print(f"Error occurred in trial {trial.number}")
        print(f"Error message: {str(e)}")
        # Save error info to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hidden_channels, num_layers, peak_lr, 
                             float('-inf'), float('-inf'), float('-inf'), float('-inf')])
        return float('-inf')  

# Create study object and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=4, n_jobs=4)  # Adjust n_trials as needed

# Get best parameters
best_params = study.best_params
best_value = study.best_value

print("\nBest parameters found:")
print(f"Hidden channels: {best_params['hidden_channels']}")
print(f"Number of layers: {best_params['num_layers']}")
print(f"Peak learning rate: {best_params['peak_lr']}")
print(f"Best test BEDROC: {best_value:.4f}")

# Run with different seeds using best parameters
seeds = [1, 2, 3]
seed_results = []

for seed in seeds:
    print(f"\nRunning with seed {seed}")
    base_config['GENERAL']['seed'] = seed
    
    try:
        # Initialize model with best params
        model = GCN_Model(
            in_channels=12,
            hidden_channels=int(best_params['hidden_channels']),
            num_layers=int(best_params['num_layers']),
        ).to(device)

        # Train model and get metrics
        if args.aug:
            aug_dataset = torch.load(f'./augment_pyg_graphs_labels/{args.dataset}_{args.split}_0.1_augment_pyg_graphs_labels.pt')
            test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = train_aug(model, dataset, aug_dataset, base_config, device, results_dir=results_dir)
        elif args.valid:
            aug_dataset = torch.load(f'./augment_valid_pyg_graphs_labels/{args.dataset}_{args.split}_0.1_augment_valid_pyg_graphs_labels.pt')
            test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = train_aug(model, dataset, aug_dataset, base_config, device, results_dir=results_dir)
        else:
            test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = train(model, dataset, base_config, device)
        
        seed_results.append({
            'seed': seed,
            'test_logAUC': test_logAUC,
            'test_EF100': test_EF100, 
            'test_DCG100': test_DCG100,
            'test_BEDROC': test_BEDROC,
            'test_EF500': test_EF500,
            'test_EF1000': test_EF1000,
            'test_DCG500': test_DCG500,
            'test_DCG1000': test_DCG1000
        })

        # Save seed results to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # Add header if file is empty
                writer.writerow([
                    'hidden_channels',
                    'num_layers',
                    'peak_lr', 
                    'test_logAUC',
                    'test_EF100',
                    'test_DCG100',
                    'test_BEDROC',
                    'test_EF500', 
                    'test_EF1000',
                    'test_DCG500',
                    'test_DCG1000',
                    'seed'
                ])
            writer.writerow([
                best_params['hidden_channels'],
                best_params['num_layers'],
                best_params['peak_lr'],
                test_logAUC,
                test_EF100, 
                test_DCG100,
                test_BEDROC,
                test_EF500,
                test_EF1000,
                test_DCG500,
                test_DCG1000,
                seed
            ])

    except Exception as e:
        print(f"Error occurred with seed {seed}")
        print(f"Error message: {str(e)}")

# Print summary statistics
if seed_results:
    seed_df = pd.DataFrame(seed_results)
    print("\nResults across seeds:")
    # Print summary statistics
    print(f"Mean test logAUC: {seed_df['test_logAUC'].mean():.4f} ± {seed_df['test_logAUC'].std():.4f}")
    print(f"Mean test EF100: {seed_df['test_EF100'].mean():.4f} ± {seed_df['test_EF100'].std():.4f}")
    print(f"Mean test DCG100: {seed_df['test_DCG100'].mean():.4f} ± {seed_df['test_DCG100'].std():.4f}")
    print(f"Mean test BEDROC: {seed_df['test_BEDROC'].mean():.4f} ± {seed_df['test_BEDROC'].std():.4f}")
    print(f"Mean test EF500: {seed_df['test_EF500'].mean():.4f} ± {seed_df['test_EF500'].std():.4f}")
    print(f"Mean test EF1000: {seed_df['test_EF1000'].mean():.4f} ± {seed_df['test_EF1000'].std():.4f}")
    print(f"Mean test DCG500: {seed_df['test_DCG500'].mean():.4f} ± {seed_df['test_DCG500'].std():.4f}")
    print(f"Mean test DCG1000: {seed_df['test_DCG1000'].mean():.4f} ± {seed_df['test_DCG1000'].std():.4f}")

    # Save summary statistics to CSV
    summary_stats = {
        'Metric': ['logAUC', 'EF100', 'DCG100', 'BEDROC', 'EF500', 'EF1000', 'DCG500', 'DCG1000'],
        'Mean': [
            seed_df['test_logAUC'].mean(),
            seed_df['test_EF100'].mean(),
            seed_df['test_DCG100'].mean(),
            seed_df['test_BEDROC'].mean(),
            seed_df['test_EF500'].mean(),
            seed_df['test_EF1000'].mean(),
            seed_df['test_DCG500'].mean(),
            seed_df['test_DCG1000'].mean()
        ],
        'Std': [
            seed_df['test_logAUC'].std(),
            seed_df['test_EF100'].std(),
            seed_df['test_DCG100'].std(),
            seed_df['test_BEDROC'].std(),
            seed_df['test_EF500'].std(),
            seed_df['test_EF1000'].std(),
            seed_df['test_DCG500'].std(),
            seed_df['test_DCG1000'].std()
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_csv = f'results/summary_stats_{dataset_name}_{split_scheme}_{timestamp}.csv'
    summary_df.to_csv(summary_csv, index=False)


