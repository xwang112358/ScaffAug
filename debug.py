import torch
from welqrate.dataset import WelQrateDataset
from welqrate.models.gnn2d.UpGIN import GIN
from welqrate.train import train
import yaml
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AID1798', help='Dataset name')
parser.add_argument('--split', type=str, default='random_cv2', help='Split scheme')
args = parser.parse_args()

# Load base config
with open('./configs/gin.yaml') as file:
    config = yaml.safe_load(file)

# Update config with dataset and split
config['DATA']['dataset_name'] = args.dataset
config['DATA']['split_scheme'] = args.split

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
print(f"Loading dataset: {args.dataset}")
dataset = WelQrateDataset(dataset_name=args.dataset, root='./welqrate_datasets', mol_repr='2dmol')

# Create output directory
results_dir = 'test_results'
os.makedirs(results_dir, exist_ok=True)

# Extract hyperparameters from config
emb_dim = 64
num_layer = 3
drop_ratio = 0.25
graph_pooling = "sum"

print("\nTraining configuration:")
print(f"Embedding dimension: {emb_dim}")
print(f"Number of layers: {num_layer}")
print(f"Dropout ratio: {drop_ratio}")
print(f"Graph pooling: {graph_pooling}")
print(f"Batch size: {config['TRAIN']['batch_size']}")
print(f"Seed: {config['GENERAL']['seed']}")

# Initialize model
model = GIN(
    num_layer=num_layer,
    emb_dim=emb_dim,
    drop_ratio=drop_ratio,
    graph_pooling=graph_pooling
).to(device)

print(f"\nModel architecture:\n{model}")
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# Train model
save_path = f"{results_dir}/{args.dataset}_{args.split}"
os.makedirs(save_path, exist_ok=True)

print("\nStarting training...")
try:
    test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = train(
        model=model, 
        config=config, 
        device=device,
        save_path=save_path,
        dataset=dataset
    )
    
    print("\nTraining completed successfully!")
    print("\nTest Results:")
    print(f"logAUC: {test_logAUC:.4f}")
    print(f"EF100: {test_EF100:.4f}")
    print(f"DCG100: {test_DCG100:.4f}")
    print(f"BEDROC: {test_BEDROC:.4f}")
    print(f"EF500: {test_EF500:.4f}")
    print(f"EF1000: {test_EF1000:.4f}")
    print(f"DCG500: {test_DCG500:.4f}")
    print(f"DCG1000: {test_DCG1000:.4f}")
    
except Exception as e:
    print(f"\nError during training: {str(e)}")
    import traceback
    traceback.print_exc()