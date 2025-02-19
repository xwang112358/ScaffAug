from welqrate.dataset import WelQrateDataset
from welqrate.models.gnn2d.GAT import GAT_Model 
import torch
from welqrate.train import train
from welqrate.train_aug import train as train_aug
import yaml
import itertools
import copy
import os
import pandas as pd
import csv
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AID1798', required=True)
parser.add_argument('--split', type=str, default='random_cv1', required=True)
parser.add_argument('--aug', action='store_true')
parser.add_argument('--valid', action='store_true')
args = parser.parse_args()

# Define hyperparameter search space for GAT
hyperparams = {
    'hidden_channels': [32, 64, 128], 
    'num_layers': [3, 4, 5],
    'peak_lr': [1e-2, 1e-3, 1e-4],
    'heads': [4, 8]
}


# Load base config
with open('./configs/gat.yaml') as file:
    base_config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = WelQrateDataset(dataset_name=args.dataset, root='./welqrate_datasets', mol_repr='2dmol')

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.aug:
    results_dir = 'results_aug'
    csv_file = f'results_aug/gat_finetuning_{args.dataset}_{args.split}_{timestamp}.csv'
elif args.valid:
    results_dir = 'results_valid_aug'
    csv_file = f'results_valid_aug/gat_finetuning_{args.dataset}_{args.split}_{timestamp}.csv'
else:
    results_dir = 'results'
    csv_file = f'results/gat_finetuning_{args.dataset}_{args.split}_{timestamp}.csv'

os.makedirs(results_dir, exist_ok=True)

# Initialize results DataFrame
results_data = []
    
# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(
    hyperparams['hidden_channels'],
    hyperparams['num_layers'],
    hyperparams['peak_lr'],
    hyperparams['heads']
))

# Get dataset name and split scheme from config
dataset_name = args.dataset
split_scheme = args.split

# Create CSV file with headers
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'hidden_channels', 'num_layers', 'peak_lr', 'heads',
        'test_logAUC', 'test_EF', 'test_DCG', 'test_BEDROC'
    ])


# Run training for each combination
for hidden_ch, n_layers, lr, n_heads in param_combinations:
    try:
        # Update config
        config = copy.deepcopy(base_config)
        config['MODEL']['hidden_channels'] = hidden_ch
        config['MODEL']['num_layers'] = n_layers
        config['MODEL']['heads'] = n_heads
        config['TRAIN']['peak_lr'] = lr
        config['DATA']['split_scheme'] = args.split
        
        # Initialize model with current params
        model = GAT_Model(
            in_channels=12,
            hidden_channels=hidden_ch,
            num_layers=n_layers,
            heads=n_heads,
        ).to(device)
        
        print(f"\nTraining with parameters:")
        print(f"Hidden channels: {hidden_ch}")
        print(f"Number of layers: {n_layers}")
        print(f"Number of heads: {n_heads}")
        print(f"Peak learning rate: {lr}")

        # Train model and get metrics
        if args.aug:
            aug_dataset = torch.load(f'./augment_pyg_graphs_labels/{args.dataset}_{args.split}_0.1_augment_pyg_graphs_labels.pt')
            test_logAUC, test_EF100, test_DCG100, test_BEDROC, _, _, _, _ = train_aug(model, dataset, aug_dataset, config, device, results_dir=results_dir)
        elif args.valid:
            aug_dataset = torch.load(f'./augment_valid_pyg_graphs_labels/{args.dataset}_{args.split}_0.1_augment_valid_pyg_graphs_labels.pt')
            test_logAUC, test_EF100, test_DCG100, test_BEDROC, _, _, _, _ = train_aug(model, dataset, aug_dataset, config, device, results_dir=results_dir)
        else:
            test_logAUC, test_EF100, test_DCG100, test_BEDROC, _, _, _, _ = train(model, dataset, config, device)
        
        # Extract metrics
        test_metrics = [
            test_logAUC,
            test_EF100,
            test_DCG100,
            test_BEDROC
        ]

        # Save results to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hidden_ch, n_layers, lr, n_heads] + test_metrics)
        
        # Store results for DataFrame
        results_data.append({
            'hidden_channels': hidden_ch,
            'num_layers': n_layers,
            'peak_lr': lr,
            'heads': n_heads,
            'test_logAUC': test_logAUC,
            'test_EF100': test_EF100,
            'test_DCG100': test_DCG100,
            'test_BEDROC': test_BEDROC
        })
        
    except Exception as e:
        print(f"Error occurred with parameters: hidden_ch={hidden_ch}, n_layers={n_layers}, lr={lr}, heads={n_heads}")
        print(f"Error message: {str(e)}")
        
        # Save error in CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hidden_ch, n_layers, lr, n_heads, 'ERROR', 'ERROR', 'ERROR', 'ERROR'])

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results_data)
if args.aug:
    final_results_csv = f'results_aug/gat_final_{args.dataset}_{args.split}_{timestamp}.csv'
elif args.valid:
    final_results_csv = f'results_valid_aug/gat_final_{args.dataset}_{args.split}_{timestamp}.csv'
else:
    final_results_csv = f'results/gat_final_{args.dataset}_{args.split}_{timestamp}.csv'

if not results_df.empty:
    # Find parameters with best test_logAUC
    best_params = results_df.loc[results_df['test_logAUC'].idxmax()]
    print("\nBest parameters found:")
    print(f"Hidden channels: {best_params['hidden_channels']}")
    print(f"Number of layers: {best_params['num_layers']}")
    print(f"Number of heads: {best_params['heads']}")
    print(f"Peak learning rate: {best_params['peak_lr']}")
    print(f"Best test logAUC: {best_params['test_logAUC']:.4f}")

    # Run with different seeds
    seeds = [1, 2, 3]
    seed_results = []

    for seed in seeds:
        print(f"\nRunning with seed {seed}")
        config['GENERAL']['seed'] = seed
        
        try:
            # Initialize model with best params
            model = GAT_Model(
                in_channels=12,
                hidden_channels=int(best_params['hidden_channels']),
                num_layers=int(best_params['num_layers']),
                heads=int(best_params['heads']),
            ).to(device)

            # Train model and get metrics
            if args.aug:
                aug_dataset = torch.load(f'./augment_pyg_graphs_labels/{args.dataset}_{args.split}_0.1_augment_pyg_graphs_labels.pt')
                test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = train_aug(model, dataset, aug_dataset, config, device, results_dir)
            elif args.valid:
                aug_dataset = torch.load(f'./augment_valid_pyg_graphs_labels/{args.dataset}_{args.split}_0.1_augment_valid_pyg_graphs_labels.pt')
                test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = train_aug(model, dataset, aug_dataset, config, device, results_dir)
            else:
                test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = train(model, dataset, config, device)
            
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
            with open(final_results_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:  # Add header if file is empty
                    writer.writerow([
                        'hidden_channels',
                        'num_layers',
                        'peak_lr',
                        'heads',
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
                    best_params['heads'],
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
        print(f"Mean test logAUC: {seed_df['test_logAUC'].mean():.4f} ± {seed_df['test_logAUC'].std():.4f}")
        print(f"Mean test EF100: {seed_df['test_EF100'].mean():.4f} ± {seed_df['test_EF100'].std():.4f}")
        print(f"Mean test DCG100: {seed_df['test_DCG100'].mean():.4f} ± {seed_df['test_DCG100'].std():.4f}")
        print(f"Mean test BEDROC: {seed_df['test_BEDROC'].mean():.4f} ± {seed_df['test_BEDROC'].std():.4f}")
        print(f"Mean test EF500: {seed_df['test_EF500'].mean():.4f} ± {seed_df['test_EF500'].std():.4f}")
        print(f"Mean test EF1000: {seed_df['test_EF1000'].mean():.4f} ± {seed_df['test_EF1000'].std():.4f}")
        print(f"Mean test DCG500: {seed_df['test_DCG500'].mean():.4f} ± {seed_df['test_DCG500'].std():.4f}")
        print(f"Mean test DCG1000: {seed_df['test_DCG1000'].mean():.4f} ± {seed_df['test_DCG1000'].std():.4f}")

