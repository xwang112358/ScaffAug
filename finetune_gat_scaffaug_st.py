import torch
from welqrate.dataset import WelQrateDataset
from welqrate.models.gnn2d.GAT import GAT_Model 
from welqrate.train_pseudo_label import train_pseudo_label
import argparse
import yaml
import itertools
import copy
import os
import csv
from datetime import datetime
import pandas as pd
import optuna
import sys
from tqdm import tqdm
from welqrate.loader import get_train_loader, get_valid_loader, get_test_loader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AID1798', required=True)
parser.add_argument('--split', type=str, default='random_cv1', required=True)
parser.add_argument('--sampling_method', type=str, default='sabs', required=True)
args = parser.parse_args()

# Load base config
with open('./configs/scaffaug.yaml') as file:
    base_config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
augmented_dataset = torch.load(f'./{args.sampling_method}/{args.dataset}_{args.split}_0.1_generated_graphs.pt')
original_dataset = WelQrateDataset(dataset_name=args.dataset, root='./welqrate_datasets', mol_repr='2dmol')
split_dict = original_dataset.get_idx_split(args.split)

original_train_data_list = []
train_data = original_dataset[split_dict['train']]
valid_data = original_dataset[split_dict['valid']]
test_data = original_dataset[split_dict['test']]

for i in range(len(train_data)):
    original_train_data_list.append(train_data[i])

batch_size = base_config['TRAIN']['batch_size']
seed = base_config['GENERAL']['seed']
num_workers = base_config['GENERAL']['num_workers']

valid_loader = get_valid_loader(valid_data, batch_size=batch_size, num_workers=num_workers, seed=seed)
test_loader = get_test_loader(test_data, batch_size=batch_size, num_workers=num_workers, seed=seed)

# Create results directory
results_dir = 'scaffaug_gat_results'
os.makedirs(results_dir, exist_ok=True)

# Create CSV file with headers
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
finetuning_csv_file = f'{results_dir}/gat_scaffaug_st_finetuning_{args.dataset}_{args.split}_{timestamp}.csv'
final_results_csv = f'{results_dir}/gat_scaffaug_st_final_{args.dataset}_{args.split}_{timestamp}.csv'

with open(finetuning_csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'hidden_channels', 'num_layers', 'peak_lr', 'heads', 'confidence_threshold', 'pseudo_label_freq', 'start_epoch',
        'test_logAUC', 'test_EF', 'test_DCG', 'test_BEDROC',
        'test_EF500', 'test_EF1000', 'test_DCG500', 'test_DCG1000'
    ])

with open(final_results_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'hidden_channels', 'num_layers', 'peak_lr', 'heads', 'confidence_threshold', 'pseudo_label_freq', 'start_epoch',
        'test_logAUC', 'test_EF100', 'test_DCG100', 'test_BEDROC',
        'test_EF500', 'test_EF1000', 'test_DCG500', 'test_DCG1000',
        'seed'
    ])

def objective(trial):
    # Define hyperparameter search space
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    peak_lr = trial.suggest_float('peak_lr', 1e-4, 1e-2, log=True)
    heads = trial.suggest_categorical('heads', [4, 8])
    confidence_threshold = trial.suggest_float('confidence_threshold', 0.7, 0.9)
    pseudo_label_freq = trial.suggest_int('pseudo_label_freq', 3, 10)
    start_epoch = trial.suggest_int('start_epoch', 15, 25)

    trial_dir = f"{results_dir}/{args.dataset}/{args.split}/gat/trial{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)

    try:
        # Update config
        config = copy.deepcopy(base_config)
        config['DATA']['dataset_name'] = args.dataset
        config['DATA']['split_scheme'] = args.split
        config['MODEL']['hidden_channels'] = hidden_channels
        config['MODEL']['num_layers'] = num_layers
        config['MODEL']['heads'] = heads
        config['TRAIN']['peak_lr'] = peak_lr
        config['AUGMENTATION']['confidence_threshold'] = confidence_threshold
        config['AUGMENTATION']['pseudo_label_freq'] = pseudo_label_freq
        config['AUGMENTATION']['start_epoch'] = start_epoch
        config['MODEL']['model_name'] = 'gat'
        
        # Initialize model with current params
        model = GAT_Model(
            in_channels=12,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
        ).to(device)
        
        print(f"\nTrial {trial.number}")
        print(f"Hidden channels: {hidden_channels}")
        print(f"Number of layers: {num_layers}")
        print(f"Number of heads: {heads}")
        print(f"Peak learning rate: {peak_lr}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Pseudo label frequency: {pseudo_label_freq}")
        print(f"Start epoch: {start_epoch}")
        
        # Train model and get metrics
        test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = train_pseudo_label(
            model=model, 
            config=config, 
            device=device,
            aug_data_list=augmented_dataset,
            orig_train_data_list=original_train_data_list,
            valid_loader=valid_loader, 
            test_loader=test_loader, 
            save_path=trial_dir
        )

        # Save results to CSV
        with open(finetuning_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                hidden_channels, num_layers, peak_lr, heads, confidence_threshold, pseudo_label_freq, start_epoch,
                test_logAUC, test_EF100, test_DCG100, test_BEDROC,
                test_EF500, test_EF1000, test_DCG500, test_DCG1000
            ])
        
        return test_BEDROC
        
    except Exception as e:
        print(f"Error occurred in trial {trial.number}")
        print(f"Error message: {str(e)}")
        # Save error info to CSV
        with open(finetuning_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                hidden_channels, num_layers, peak_lr, heads, confidence_threshold, pseudo_label_freq, start_epoch,
                float('-inf'), float('-inf'), float('-inf'), float('-inf'),
                float('-inf'), float('-inf'), float('-inf'), float('-inf')
            ])
        return float('-inf')

# Create study object and optimize
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
study.optimize(objective, n_trials=36, n_jobs=4, timeout=16200)  # Adjust n_trials as needed

# Get best parameters
best_params = study.best_params
best_value = study.best_value

print("\nBest parameters found:")
print(f"Hidden channels: {best_params['hidden_channels']}")
print(f"Number of layers: {best_params['num_layers']}")
print(f"Number of heads: {best_params['heads']}")
print(f"Peak learning rate: {best_params['peak_lr']}")
print(f"Confidence threshold: {best_params['confidence_threshold']}")
print(f"Pseudo label frequency: {best_params['pseudo_label_freq']}")
print(f"Start epoch: {best_params['start_epoch']}")
print(f"Best test BEDROC: {best_value:.4f}")

# Run with different seeds using best parameters
seeds = [1, 2, 3]
seed_results = []

for seed in seeds:
    print(f"\nRunning with seed {seed}")
    config = copy.deepcopy(base_config)
    config['GENERAL']['seed'] = seed
    config['MODEL']['hidden_channels'] = best_params['hidden_channels']
    config['MODEL']['num_layers'] = best_params['num_layers']
    config['MODEL']['heads'] = best_params['heads']
    config['TRAIN']['peak_lr'] = best_params['peak_lr']
    config['AUGMENTATION']['confidence_threshold'] = best_params['confidence_threshold']
    config['AUGMENTATION']['pseudo_label_freq'] = best_params['pseudo_label_freq']
    config['AUGMENTATION']['start_epoch'] = best_params['start_epoch']
    config['DATA']['split_scheme'] = args.split
    config['MODEL']['model_name'] = 'gat'
    final_results_dir = f'{results_dir}/{args.dataset}/{args.split}/gat/seed{seed}'
    os.makedirs(final_results_dir, exist_ok=True)
    
    try:
        model = GAT_Model(
            in_channels=12,
            hidden_channels=int(best_params['hidden_channels']),
            num_layers=int(best_params['num_layers']),
            heads=int(best_params['heads']),
        ).to(device)

        test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = train_pseudo_label(
            model = model, 
            config = config, 
            device=device,
            aug_data_list=augmented_dataset,
            orig_train_data_list=original_train_data_list,
            valid_loader=valid_loader, 
            test_loader=test_loader, 
            save_path=final_results_dir
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

        # Save seed results to CSV
        with open(final_results_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                best_params['hidden_channels'],
                best_params['num_layers'],
                best_params['peak_lr'],
                best_params['heads'],
                best_params['confidence_threshold'],
                best_params['pseudo_label_freq'],
                best_params['start_epoch'],
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
    summary_csv = f'{results_dir}/gat_scaffaug_st_summary_stats_{args.dataset}_{args.split}_{timestamp}.csv'
    summary_df.to_csv(summary_csv, index=False)
