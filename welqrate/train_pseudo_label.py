from tqdm import tqdm
import numpy as np
import torch
from torch_scatter import scatter_add
import random
import os
from datetime import datetime
from welqrate.loader import get_train_loader, get_test_loader, get_valid_loader
from welqrate.scheduler import get_scheduler, get_lr
from welqrate.utils.evaluation import calculate_logAUC, cal_EF, cal_DCG, cal_BEDROC_score
from welqrate.utils.rank_prediction import rank_prediction
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
import yaml
from torch_geometric.loader import DataLoader
from torch.nn import functional as F


def get_pseudo_labels(model, aug_loader, device, confidence_threshold=0.6):
    """Generate pseudo labels for augmented data with confidence thresholding"""
    model.eval()
    pseudo_labels = []
    confident_mask = []
    
    with torch.no_grad():
        for aug_batch in aug_loader:
            aug_batch.to(device)
            logits = model(aug_batch)
            probs = torch.sigmoid(logits)
            
            # Consider both high confidence positive and negative predictions
            confident = (probs > confidence_threshold) | (probs < (1 - confidence_threshold))
            pseudo_label = (probs > 0.5).float()
            
            pseudo_labels.append(pseudo_label)
            confident_mask.append(confident)
    
    # Concatenate and move to CPU before converting to numpy
    pseudo_labels = torch.cat(pseudo_labels)
    confident_mask = torch.cat(confident_mask)
    
    # Print statistics about confident predictions
    total_confident = confident_mask.cpu().sum().item()
    confident_ones = (confident_mask & (pseudo_labels > 0.5)).cpu().sum().item()
    confident_zeros = (confident_mask & (pseudo_labels <= 0.5)).cpu().sum().item()
    print(f'Number of confident predictions: {total_confident}')
    print(f'Number of confident positive predictions: {confident_ones}')
    print(f'Number of confident negative predictions: {confident_zeros}')
            
    return pseudo_labels, confident_mask

def get_train_loss(model, loader, aug_loader, optimizer, scheduler, device, loss_fn, aug_weight=0.3, confidence_threshold=0.6):
    """Modified training loop with pseudo labeling and combined loss"""
    model.train()
    loss_list = []
    aug_loss_list = []

    # Generate pseudo labels for augmented data first
    pseudo_labels, confident_mask = get_pseudo_labels(model, aug_loader, device, confidence_threshold)
    
    # Combine training on original and augmented data
    for (batch, aug_batch), i in zip(zip(loader, aug_loader), range(len(loader))):
        optimizer.zero_grad()
        
        # Forward pass and loss computation on original data
        batch.to(device)
        y_pred = model(batch)
        orig_loss = loss_fn(y_pred.view(-1), batch.y.view(-1).float())
        loss_list.append(orig_loss.item())

        # Forward pass and loss computation on augmented data
        aug_batch.to(device)
        batch_size = aug_batch.y.size(0)
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        # Get confident pseudo labels for this batch
        batch_confident = confident_mask[start_idx:end_idx]
        
        # Only compute augmented loss if there are confident predictions
        if batch_confident.any():
            aug_pred = model(aug_batch)
            batch_pseudo_labels = pseudo_labels[start_idx:end_idx]
            
            # Ensure all tensors have matching dimensions
            aug_pred = aug_pred.view(-1).to(device)
            batch_pseudo_labels = batch_pseudo_labels.view(-1).to(device)
            batch_confident = batch_confident.to(device)
            
            # Compute loss only for confident predictions
            aug_loss = loss_fn(
                aug_pred[batch_confident.squeeze()].view(-1),
                batch_pseudo_labels[batch_confident.squeeze()].view(-1)
            )
            aug_loss_list.append(aug_loss.item())
        else:
            aug_loss = torch.tensor(0.0, device=device)

        # Combine losses and perform single backward pass
        total_loss = (1 - aug_weight) * orig_loss + aug_weight * aug_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = np.mean(loss_list)
    avg_aug_loss = np.mean(aug_loss_list) if aug_loss_list else 0
    return avg_loss, avg_aug_loss


def train(model, dataset, aug_dataset, config, device, train_eval=False):
    # train params: batch_size, num_epochs, weight_decay, peark_lr, ...
    # split_scheme --> dataset --> loaders 
    # load train info
    batch_size = int(config['TRAIN']['batch_size'])
    num_epochs = int(config['TRAIN']['num_epochs'])
    num_workers = int(config['GENERAL']['num_workers'])
    seed = int(config['GENERAL']['seed'])
    weight_decay = float(config['TRAIN']['weight_decay'])
    early_stopping_limit = int(config['TRAIN']['early_stop'])
    split_scheme = config['DATA']['split_scheme']
    aug_weight = float(config['AUGMENTATION']['aug_weight'])
    confidence_threshold = float(config['AUGMENTATION']['confidence_threshold'])
    loss_fn = BCEWithLogitsLoss()
    
    # load dataset info
    dataset_name = dataset.name
    root = dataset.root
    mol_repr = dataset.mol_repr
    
    # create loader
    split_dict = dataset.get_idx_split(split_scheme)
    train_loader = get_train_loader(dataset[split_dict['train']], batch_size, num_workers, seed)
    aug_train_loader = DataLoader(aug_dataset, batch_size=batch_size)
    valid_loader = get_valid_loader(dataset[split_dict['valid']], batch_size, num_workers, seed)
    test_loader = get_test_loader(dataset[split_dict['test']], batch_size, num_workers, seed) 

    # load model info
    model_name = config['MODEL']['model_name']

    # load optimizer and scheduler
    optimizer = AdamW(model.parameters(), weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, config, train_loader)
    
    print('\n' + '=' * 10 + f"Training {model} on {dataset_name}'s {split_scheme} split" '\n' + '=' * 10 )
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Modified base path initialization with versioning
    base_path = f'./results/{dataset_name}_aug_pseudo_label/{split_scheme}/{model_name}0'
    version = 0
    while os.path.exists(base_path):
        version += 1
        base_path = f'./results/{dataset_name}_aug_pseudo_label/{split_scheme}/{model_name}{version}'
    
    model_save_path = os.path.join(base_path, f'{model_name}.pt')
    log_save_path = os.path.join(base_path, f'train.log')
    metrics_save_path = os.path.join(base_path, f'test_results.txt')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # save config
    with open(os.path.join(base_path, f'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    
    best_epoch = 0
    best_valid_logAUC = -1
    early_stopping_counter = 0
    print(f'Training with early stopping limit of {early_stopping_limit} epochs')
    
    with open(log_save_path, 'w+') as out_file:
        for epoch in range(num_epochs):
            train_loss, aug_loss = get_train_loss(model, train_loader, aug_train_loader, 
                                                optimizer, scheduler, device, loss_fn,
                                                aug_weight=aug_weight)
            
            lr = get_lr(optimizer)
            
            if train_eval:
                train_logAUC, train_EF, train_DCG, train_BEDROC = get_test_metrics(model, train_loader, device)
                print(f'current_epoch={epoch} train_loss={train_loss:.4f} aug_loss={aug_loss:.4f} lr={lr}')
                print(f'train_logAUC={train_logAUC:.4f} train_EF={train_EF:.4f} train_DCG={train_DCG:.4f} train_BEDROC={train_BEDROC:.4f}')
                out_file.write(f'Epoch:{epoch}\tloss={train_loss}\tlogAUC={train_logAUC}\tEF={train_EF}\tDCG={train_DCG}\tBEDROC={train_BEDROC}\tlr={lr}\t\n')
            
            else:
                print(f'current_epoch={epoch} train_loss={train_loss:.4f} lr={lr}')
                out_file.write(f'Epoch:{epoch}\tloss={train_loss}\tlr={lr}\t\n')
                
            valid_logAUC, valid_EF, valid_DCG, valid_BEDROC = get_test_metrics(model, valid_loader, device, type='valid', save_per_molecule_pred=True, save_path=base_path)  
            print(f'valid_logAUC={valid_logAUC:.4f} valid_EF={valid_EF:.4f} valid_DCG={valid_DCG:.4f} valid_BEDROC={valid_BEDROC:.4f}')
            out_file.write(f'Epoch:{epoch}\tlogAUC={valid_logAUC}\tEF={valid_EF}\tDCG={valid_DCG}\tBEDROC={valid_BEDROC}\t\n')  
            
            if valid_logAUC > best_valid_logAUC: 
                best_valid_logAUC = valid_logAUC
                best_epoch = epoch
                torch.save({'model': model.state_dict(),
                            'epoch': epoch}, model_save_path)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_limit:
                print(f'Early stopping at epoch {epoch}')
                break
        print(f'Training finished')
        print(f'Best epoch: {best_epoch} with valid logAUC: {best_valid_logAUC:.4f}')
        
    # testing the model
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path)['model'])
        print(f'Best Model loeaded from {model_save_path}')
    else:
        raise Exception(f'Model not found at {model_save_path}')
   
    print('Testing ...')
    test_logAUC, test_EF, test_DCG, test_BEDROC = get_test_metrics(model, test_loader, device, 
                                                                   save_per_molecule_pred=True,
                                                                   save_path=base_path)
    print(f'{model_name} at epoch {best_epoch} test logAUC: {test_logAUC:.4f} test EF: {test_EF:.4f} test DCG: {test_DCG:.4f} test BEDROC: {test_BEDROC:.4f}')
    with open(metrics_save_path, 'w+') as result_file:
        result_file.write(f'logAUC={test_logAUC}\tEF={test_EF}\tDCG={test_DCG}\tBEDROC={test_BEDROC}\t\n')
    
    return test_logAUC, test_EF, test_DCG, test_BEDROC
    

def get_test_metrics(model, loader, device, type = 'test', save_per_molecule_pred=False, save_path=None):
    model.eval()

    all_pred_y = []
    all_true_y = []

    for i, batch in enumerate(tqdm(loader)):
        batch.to(device)
        pred_y = model(batch).cpu().view(-1).detach().numpy()
        true_y = batch.y.view(-1).cpu().numpy()
        for j, _ in enumerate(pred_y):
            all_pred_y.append(pred_y[j])
            all_true_y.append(true_y[j])
    
    if save_per_molecule_pred and save_path is not None:
        filename = os.path.join(save_path, f'per_molecule_pred_of_{type}_set.txt')
        with open(filename, 'w') as out_file:
            for k, _ in enumerate(all_pred_y):
                out_file.write(f'{all_pred_y[k]}\ty={all_true_y[k]}\n')
        
        # rank prediction
        with open(filename, 'r') as f:
            data = [(float(line.split('\t')[0]), line.split('\t')[1] ) for line in f.readlines()]
        ranked_data = sorted(data, key=lambda x: x[0], reverse=True)
        with open(os.path.join(save_path, f'ranked_mol_score_{type}.txt'), 'w') as f:
            for i, (score, label) in enumerate(ranked_data):
                f.write(f"{i}\t{score}\t{label}")

    all_pred_y = np.array(all_pred_y)
    all_true_y = np.array(all_true_y)
    logAUC = calculate_logAUC(all_true_y, all_pred_y)
    EF = cal_EF(all_true_y, all_pred_y, 100)
    DCG = cal_DCG(all_true_y, all_pred_y, 100)
    BEDROC = cal_BEDROC_score(all_true_y, all_pred_y)
    return logAUC, EF, DCG, BEDROC


