import torch
from welqrate.dataset import WelQrateDataset
from welqrate.models.gnn2d.GCN import GCN_Model 
from welqrate.train_pseudo_label import train as train_pseudo_label
import argparse
import yaml
import itertools
import copy
import os
import csv
from datetime import datetime
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AID1798')
parser.add_argument('--split', type=str, default='random_cv1')
# parser.add_argument('--ratio', type=float, default=0.1, required=True) 
args = parser.parse_args()

# Define hyperparameter search space
hyperparams = {
    'hidden_channels': [32, 64, 128],
    'num_layers': [2, 3, 4],
    'peak_lr': [1e-2, 1e-3, 1e-4],
    'confidence_threshold': [0.7, 0.8]
}

# Load base config
with open('./configs/scaffaug.yaml') as file:
    base_config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
augmented_dataset = torch.load(f'./augment_pyg_datasets/{args.dataset}_{args.split}_0.1_generated_pyg_graphs.pt')
original_dataset = WelQrateDataset(dataset_name=args.dataset, root='./welqrate_datasets', mol_repr='2dmol')

os.makedirs('scaffaug_results', exist_ok=True)

# Create results directory and CSV file
results_data = []

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(
    hyperparams['hidden_channels'],
    hyperparams['num_layers'],
    hyperparams['peak_lr'],
    hyperparams['confidence_threshold']
))


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f'scaffaug_results/scaffaug_st_gcn_finetuning_{args.dataset}_{args.split}_{timestamp}.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'hidden_channels', 'num_layers', 'peak_lr', 'confidence_threshold',
        'test_logAUC', 'test_EF', 'test_DCG', 'test_BEDROC'
    ])

best_params_file = f'scaffaug_results/best_parameters_{args.dataset}_{args.split}_{timestamp}.txt'

# Run training for each combination
for hidden_ch, n_layers, lr, confidence_threshold in param_combinations:
    
    # Update config
    try:
        config = copy.deepcopy(base_config)
        config['MODEL']['hidden_channels'] = hidden_ch
        config['MODEL']['num_layers'] = n_layers
        config['TRAIN']['peak_lr'] = lr
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
        print(f"Confidence threshold: {confidence_threshold}")
        
        # Train model and get metrics
        test_logAUC, test_EF100, test_DCG100, test_BEDROC, _, _, _, _ = train_pseudo_label(
            model, original_dataset, augmented_dataset, config, device
        )

        test_metrics = [
            test_logAUC,
            test_EF100,
            test_DCG100,
            test_BEDROC
        ]

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hidden_ch, n_layers, lr, confidence_threshold] + test_metrics)
        
        results_data.append({
            'hidden_channels': hidden_ch,
            'num_layers': n_layers,
            'peak_lr': lr,
            'confidence_threshold': confidence_threshold,
            'test_logAUC': test_logAUC,
            'test_EF': test_EF100,
            'test_DCG': test_DCG100,
            'test_BEDROC': test_BEDROC
        })
    except Exception as e:
        print(f"Error occurred with parameters: {hidden_ch}, {n_layers}, {lr}, {confidence_threshold}")
        print(f"Error message: {str(e)}")
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hidden_ch, n_layers, lr, confidence_threshold] + ['ERROR'] * 8)

results_df = pd.DataFrame(results_data)
final_results_csv = f'scaffaug_results/scaffaug_st_gcn_finetuning_{args.dataset}_{args.split}_{timestamp}.csv'

if not results_df.empty:
    best_params = results_df.loc[results_df['test_logAUC'].idxmax()]
    print("\nBest parameters found:")
    print(f"Hidden channels: {best_params['hidden_channels']}")
    print(f"Number of layers: {best_params['num_layers']}")
    print(f"Peak learning rate: {best_params['peak_lr']}")
    print(f"Confidence threshold: {best_params['confidence_threshold']}")
    print(f"Best test logAUC: {best_params['test_logAUC']:.4f}")

    seeds = [1, 2, 3]
    seed_results = []

    for seed in seeds:
        print(f"\nRunning with seed {seed}")
        config['GENERAL']['seed'] = seed
        
        try:
            model = GCN_Model(
                in_channels=12,
                hidden_channels=int(best_params['hidden_channels']),
                num_layers=int(best_params['num_layers']),
            ).to(device)

            (test_logAUC, test_EF100, test_DCG100, test_BEDROC,
             test_EF500, test_EF1000, test_DCG500, test_DCG1000) = train_pseudo_label(
                model, original_dataset, augmented_dataset, config, device
            )

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

            with open(final_results_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow([
                        'hidden_channels',
                        'num_layers',
                        'peak_lr',
                        'confidence_threshold',
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
                    best_params['confidence_threshold'],
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
        
        
    
        
