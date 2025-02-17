from tqdm import tqdm
import numpy as np
import torch
from torch_scatter import scatter_add
import random
import os
from datetime import datetime
from welqrate.loader import get_train_loader, get_test_loader, get_valid_loader, get_self_train_loader
from welqrate.scheduler import get_scheduler, get_lr
from welqrate.utils.evaluation import calculate_logAUC, cal_EF, cal_DCG, cal_BEDROC_score
from welqrate.utils.rank_prediction import rank_prediction
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
import yaml
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from welqrate.utils.test import get_test_metrics
from torch_geometric.data import InMemoryDataset, Data


# -----------------Self-Training Helper Classes -----------------
class AugmentedDataset(InMemoryDataset):
    """PyG dataset for storing augmented data with pseudo-labels"""
    def __init__(self, data_list):
        super().__init__()
        self.data, self.slices = self.collate(data_list)
        # not sure initialize 0 is correct 
        # should be -1 or None?
        # remember y is torch.tensor([1], dtype=int32)
        self.pseudo_labels = torch.zeros(len(self), dtype=torch.float)
        self.confidence_mask = torch.zeros(len(self), dtype=torch.bool)
        
    def update_labels(self, indices, labels, mask):
        """Update pseudo-labels in-place"""
        self.pseudo_labels[indices] = labels
        self.confidence_mask[indices] = mask

class CombinedDataset(InMemoryDataset):
    """Virtual dataset combining original labeled data and confident pseudo-labels"""
    def __init__(self, orig_dataset, aug_dataset):
        super().__init__()
        # Store the full datasets
        self.orig_dataset = orig_dataset  # This should be the full training dataset
        self.aug_dataset = aug_dataset
        self.permanent_confident_indices = set()
        self._update_mapping()
    
    def _update_mapping(self):
        """Refresh index mapping when pseudo-labels change"""
        new_confident_mask = self.aug_dataset.confidence_mask.clone()
        for idx in self.permanent_confident_indices:
            new_confident_mask[idx] = False
        
        self.confident_indices = torch.where(new_confident_mask)[0]
        self.permanent_confident_indices.update(self.confident_indices.tolist())
    
    def __len__(self):
        return len(self.orig_dataset) + len(self.permanent_confident_indices)

    def get(self, idx):
        # If index is within original dataset range
        if idx < len(self.orig_dataset):
            return self.orig_dataset[idx]  # Return the actual indexed item
        
        # For augmented data
        aug_idx = idx - len(self.orig_dataset)
        permanent_idx_list = sorted(list(self.permanent_confident_indices))
        if aug_idx >= len(permanent_idx_list):
            raise IndexError(f"Index {idx} out of range for combined dataset of size {len(self)}")
            
        aug_data_idx = permanent_idx_list[aug_idx]
        data = self.aug_dataset[aug_data_idx].clone()
        label_value = self.aug_dataset.pseudo_labels[aug_data_idx]
        data.y = torch.tensor(label_value, dtype=torch.int32)
        if hasattr(self.orig_dataset[0].y, 'shape'):
            data.y = data.y.reshape(self.orig_dataset[0].y.shape)
        return data

def update_pseudo_labels(model, aug_dataset, device, threshold, batch_size):
    """Batch update of pseudo-labels for augmented dataset"""
    model.eval()
    aug_loader = DataLoader(aug_dataset, batch_size, shuffle=False)
    
    all_indices, all_probs = [], []
    with torch.no_grad():
        for i, batch in enumerate(aug_loader):
            batch = batch.to(device)
            logits = model(batch).squeeze()
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            # Calculate correct indices based on batch size
            batch_start_idx = i * batch_size
            batch_indices = torch.arange(len(batch), dtype=torch.long) + batch_start_idx
            all_indices.append(batch_indices)
    
    probs = torch.cat(all_probs)
    indices = torch.cat(all_indices)
    confidence = 2 * torch.abs(probs - 0.5)
    
    # Add safety check
    assert indices.max() < len(aug_dataset), f"Generated index {indices.max()} >= dataset size {len(aug_dataset)}"
    
    aug_dataset.update_labels(
        indices=indices,
        labels=(probs > 0.5).float(),
        mask=confidence > threshold
    )


# ----------------- Helper Functions -----------------

def create_save_path(config, dataset_name, save_path):
    """Create versioned save directory"""
    base_path = os.path.join(
        save_path,
        f'{dataset_name}_self_train/'
        f"{config['DATA']['split_scheme']}/"
        f"{config['MODEL']['model_name']}0"
    )
    version = 0
    while os.path.exists(base_path):
        version += 1
        base_path = base_path[:-1] + str(version)
    os.makedirs(base_path, exist_ok=True)
    return base_path

def log_metrics(epoch, loss, metrics, log_file, optimizer):
    """Unified logging function"""
    log_str = (
        f'Epoch:{epoch}\tloss={loss:.4f}\t'
        f'logAUC={metrics[0]:.4f}\tEF={metrics[1]:.4f}\t'
        f'DCG={metrics[2]:.4f}\tBEDROC={metrics[3]:.4f}\t'
        f'lr={get_lr(optimizer):.6f}\n'
    )
    log_file.write(log_str)

def train_epoch(model, loader, optimizer, scheduler, device, loss_fn):
    """Single training epoch"""
    model.train()
    total_loss = 0
    
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        y_pred = model(batch).view(-1, 1)
        y_true = batch.y.float().view(-1, 1)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def save_test_results(metrics, path):
    """Save final test metrics"""
    with open(path, 'w') as f:
        f.write(
            f'logAUC={metrics[0]}\tEF100={metrics[1]}\t'
            f'DCG100={metrics[2]}\tBEDROC={metrics[3]}\t'
            f'EF500={metrics[4]}\tEF1000={metrics[5]}\t'
            f'DCG500={metrics[6]}\tDCG1000={metrics[7]}\n'
        )


# ----------------- Self-Training Loop -----------------

def train(model, orig_dataset, aug_dataset, config, device, train_eval=False,
          save_path='./scaffaug_results'):
    print(f"self training")
    # Configuration parameters
    batch_size = int(config['TRAIN']['batch_size'])
    num_epochs = int(config['TRAIN']['num_epochs'])
    confidence_threshold = float(config['AUGMENTATION']['confidence_threshold'])
    update_freq = 3  # Update pseudo-labels every 3 epochs

    if not isinstance(aug_dataset, AugmentedDataset):
        aug_dataset = AugmentedDataset(aug_dataset)

    split_dict = orig_dataset.get_idx_split(config['DATA']['split_scheme'])
    combined_dataset = CombinedDataset(orig_dataset[split_dict['train']], aug_dataset)

    train_loader = get_self_train_loader(
        combined_dataset,
        batch_size,
        int(config['GENERAL']['num_workers']),
        int(config['GENERAL']['seed'])
    )

    valid_loader = get_valid_loader(orig_dataset[split_dict['valid']], batch_size,
        int(config['GENERAL']['num_workers']),
        int(config['GENERAL']['seed'])
    )
    
    optimizer = AdamW(model.parameters(), 
                     weight_decay=float(config['TRAIN']['weight_decay']))
    scheduler = get_scheduler(optimizer, config, train_loader)

    best_valid_logAUC = -1
    early_stopping_counter = 0
    base_path = create_save_path(config, orig_dataset.name, save_path)

    with open(os.path.join(base_path, 'train.log'), 'w') as log_file:
        for epoch in range(num_epochs):
            if epoch % update_freq == 0 and epoch != 0:
                update_pseudo_labels(model, aug_dataset, device, 
                                    confidence_threshold, batch_size)
                combined_dataset._update_mapping()
                train_loader = get_self_train_loader(
                    combined_dataset,
                    batch_size,
                    int(config['GENERAL']['num_workers']),
                    int(config['GENERAL']['seed'])
                )

            train_loss = train_epoch(
                model, train_loader, optimizer, 
                scheduler, device, loss_fn=BCEWithLogitsLoss()
            )   

            # Validation and logging
            valid_metrics = get_test_metrics(model, valid_loader, device)
            log_metrics(epoch, train_loss, valid_metrics, log_file, optimizer)

            # Early stopping check
            if valid_metrics[0] > best_valid_logAUC:
                best_valid_logAUC = valid_metrics[0]
                torch.save({'model': model.state_dict(), 'epoch': epoch},
                          os.path.join(base_path, 'model.pt'))
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= int(config['TRAIN']['early_stop']):
                    break

    # Final testing
    model.load_state_dict(torch.load(os.path.join(base_path, 'model.pt'))['model'])
    test_metrics = get_test_metrics(
        model, 
        get_test_loader(orig_dataset[split_dict['test']], batch_size,
                       int(config['GENERAL']['num_workers']),
                       int(config['GENERAL']['seed'])),
        device,
        extra_metrics=True
    )
    save_test_results(test_metrics, os.path.join(base_path, 'test_results.txt'))
    
    return test_metrics