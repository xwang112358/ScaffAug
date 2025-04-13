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
from welqrate.utils.test import get_test_metrics

def get_train_loss(model, 
                   all_loader, 
                   optimizer, 
                   scheduler, 
                   device, 
                   loss_fn):
    
    model.train()
    loss_list = []

    for i, batch in enumerate(all_loader):
        batch.to(device)
        # assert batch.edge_index.max() < batch.x.size(0), f"Edge index {batch.edge_index.max()} exceeds number of nodes"
        y_pred = model(batch)
        
        loss= loss_fn(y_pred.view(-1), batch.y.view(-1).float())
            
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    loss = np.mean(loss_list)

    return loss

def get_pseudo_labels(model, 
                      aug_loader, 
                      device, 
                      confidence_threshold):
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
            
            # Make sure these tensors are on the same device as the model
            pseudo_labels.append(pseudo_label.to(device))
            confident_mask.append(confident.to(device))
    
    # Concatenate and keep on device
    pseudo_labels = torch.cat(pseudo_labels).to(device)
    confident_mask = torch.cat(confident_mask).to(device)
    
    # Print statistics about confident predictions
    total_confident = confident_mask.sum().item()
    confident_ones = (confident_mask & (pseudo_labels > 0.5)).sum().item()
    confident_zeros = (confident_mask & (pseudo_labels <= 0.5)).sum().item()
    print(f'Number of confident predictions: {total_confident}')
    print(f'Number of confident positive predictions: {confident_ones}')
    print(f'Number of confident negative predictions: {confident_zeros}')

    statistics = {
        'total_confident': total_confident,
        'confident_actives': confident_ones,
        'confident_inactives': confident_zeros
    }
            
    return pseudo_labels, confident_mask, statistics

def train_pseudo_label(model, 
                       config, 
                       device, 
                       aug_data_list,
                       orig_train_data_list,
                       valid_loader,
                       test_loader,
                       save_path=None
                       ):

    # load train info
    batch_size = int(config['TRAIN']['batch_size'])
    num_epochs = int(config['TRAIN']['num_epochs'])
    num_workers = int(config['GENERAL']['num_workers'])
    seed = int(config['GENERAL']['seed'])
    weight_decay = float(config['TRAIN']['weight_decay'])
    early_stopping_limit = int(config['TRAIN']['early_stop'])
    split_scheme = config['DATA']['split_scheme']
    dataset_name = config['DATA']['dataset_name']
    # aug_weight = float(config['AUGMENTATION']['aug_weight'])
    start_epoch = int(config['AUGMENTATION']['start_epoch'])
    confidence_threshold = float(config['AUGMENTATION']['confidence_threshold'])
    pseudo_label_freq = int(config['AUGMENTATION']['pseudo_label_freq'])
    model_name = config['MODEL']['model_name']

    loss_fn = BCEWithLogitsLoss()
    num_orig_train_active = sum(1 for data in orig_train_data_list if data.y == 1)
    num_orig_train_inactive = len(orig_train_data_list) - num_orig_train_active
    precomputed_orig_train_stats = {
        'num_train_active': num_orig_train_active,
        'num_train_inactive': num_orig_train_inactive
    }

    # Create loaders if not provided
    orig_train_loader = get_train_loader(orig_train_data_list, batch_size, num_workers, seed, precomputed_orig_train_stats)
    aug_train_loader = DataLoader(aug_data_list, batch_size=batch_size)

    # load optimizer and scheduler
    optimizer = AdamW(model.parameters(), weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, config, len(orig_train_data_list))
    
    print('\n' + '=' * 10 + f"Training {model} on {dataset_name}'s {split_scheme} split" '\n' + '=' * 10 )
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    base_path = save_path
    model_save_path = os.path.join(base_path, f'{model_name}.pt')
    log_save_path = os.path.join(base_path, f'train.log')
    metrics_save_path = os.path.join(base_path, f'test_results.txt')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # save config
    with open(os.path.join(base_path, f'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    best_epoch = 0
    best_valid_BEDROC = -1
    early_stopping_counter = 0
    print(f'Training with early stopping limit of {early_stopping_limit} epochs')

    # aug_data_loader = DataLoader(aug_data_list, batch_size=batch_size)
    
    # Initialize augmented_train_loader with original data to avoid None error
    augmented_train_loader = None
    
    with open(log_save_path, 'w+') as out_file:
        for epoch in tqdm(range(num_epochs), desc='Training'):
            if epoch < start_epoch:
                train_loss = get_train_loss(model=model, 
                                            all_loader=orig_train_loader, 
                                            optimizer=optimizer, 
                                            scheduler=scheduler, 
                                            device=device, 
                                            loss_fn=loss_fn)
                lr = get_lr(optimizer)
                print(f'current_epoch={epoch} train_loss={train_loss:.4f} lr={lr}')
                out_file.write(f'Epoch:{epoch}\t loss={train_loss}\tlr={lr}\t\n')

            elif epoch >=start_epoch and epoch % pseudo_label_freq == 0:
                # get pseudo labels
                pseudo_labels, confident_mask, pseudo_label_statistics = get_pseudo_labels(model=model, 
                                                                                           aug_loader=aug_train_loader, 
                                                                                           device=device, 
                                                                                           confidence_threshold=confidence_threshold)
                num_confident = pseudo_label_statistics['total_confident']
                num_actives = pseudo_label_statistics['confident_actives']
                num_inactives = pseudo_label_statistics['confident_inactives']

                aug_train_stats = {
                    'num_train_active': num_actives + num_orig_train_active,
                    'num_train_inactive': num_inactives + num_orig_train_inactive
                }
                
                # Save pseudo-label statistics to a file
                stats_save_path = os.path.join(base_path, 'pseudo_label_statistics.txt')
                with open(stats_save_path, 'a+') as stats_file:
                    stats_file.seek(0)
                    if not stats_file.read(1):
                        stats_file.write("Epoch\tTotal_Confident\tConfident_Actives\tConfident_Inactives\n")
                    stats_file.write(f"{epoch}\t{num_confident}\t{num_actives}\t{num_inactives}\n")
            
                # get confident augmented data
                confident_indices = torch.where(confident_mask)[0].cpu().numpy()
                confident_aug_data = [aug_data_list[i] for i in confident_indices]
                for i in range(len(confident_aug_data)):
                    # Make sure to detach and move to CPU before assignment
                    confident_aug_data[i].y = pseudo_labels[confident_indices[i]].detach().cpu()
                # confident_aug_data_loader = DataLoader(confident_aug_data, batch_size=batch_size)

                augmented_train_data_list = orig_train_data_list + confident_aug_data
                augmented_train_loader = get_train_loader(train_dataset = augmented_train_data_list, 
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          seed=seed,
                                                          precomputed_stats=aug_train_stats)
                
                train_loss = get_train_loss(model=model, 
                                            all_loader=augmented_train_loader, 
                                            optimizer=optimizer, 
                                            scheduler=scheduler, 
                                            device=device, 
                                            loss_fn=loss_fn,
                                            )
                lr = get_lr(optimizer)
                print(f'current_epoch={epoch} all_train_loss={train_loss:.4f} lr={lr}')
                out_file.write(f'Epoch:{epoch}\t all_loss={train_loss}\tlr={lr}\t\n')
            else:
                # Check if augmented_train_loader is None and use orig_train_loader as fallback
                if augmented_train_loader is None:
                    augmented_train_loader = orig_train_loader
                    
                train_loss = get_train_loss(model=model, 
                                            all_loader=augmented_train_loader, 
                                            optimizer=optimizer, 
                                            scheduler=scheduler, 
                                            device=device, 
                                            loss_fn=loss_fn,
                                            )
                lr = get_lr(optimizer)
                print(f'current_epoch={epoch} all_train_loss={train_loss:.4f} lr={lr}')
                out_file.write(f'Epoch:{epoch}\t all_loss={train_loss}\tlr={lr}\t\n')
                    
            
            valid_logAUC, valid_EF100, valid_DCG100, valid_BEDROC = get_test_metrics(model=model, 
                                                                                     loader=valid_loader, 
                                                                                     device=device, 
                                                                                     type='valid', 
                                                                                     save_per_molecule_pred=True, 
                                                                                     save_path=base_path)  
            
            print(f'valid_logAUC={valid_logAUC:.4f} valid_EF100={valid_EF100:.4f} valid_DCG100={valid_DCG100:.4f} valid_BEDROC={valid_BEDROC:.4f}')
            out_file.write(f'Epoch:{epoch}\tlogAUC={valid_logAUC}\tEF100={valid_EF100}\tDCG100={valid_DCG100}\tBEDROC={valid_BEDROC}\t\n')  
            
            if valid_BEDROC > best_valid_BEDROC:
                best_valid_BEDROC = valid_BEDROC
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
        print(f'Best epoch: {best_epoch} with valid BEDROC: {best_valid_BEDROC:.4f}')
        
    # testing the model
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path)['model'])
        print(f'Best Model loeaded from {model_save_path}')
    else:
        raise Exception(f'Model not found at {model_save_path}')
   
    print('Testing ...')
    test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = get_test_metrics(model=model, 
                                                                                                                             loader=test_loader, 
                                                                                                                             device=device, 
                                                                                                                             save_per_molecule_pred=True,
                                                                                                                             save_path=base_path,
                                                                                                                             extra_metrics=True)
    print(f'{model_name} at epoch {best_epoch} test logAUC: {test_logAUC:.4f} test EF: {test_EF100:.4f} test DCG: {test_DCG100:.4f} test BEDROC: {test_BEDROC:.4f}')
    with open(metrics_save_path, 'w+') as result_file:
        result_file.write(f'logAUC={test_logAUC}\tEF100={test_EF100}\tDCG100={test_DCG100}\tBEDROC={test_BEDROC}\tEF500={test_EF500}\tEF1000={test_EF1000}\tDCG500={test_DCG500}\tDCG1000={test_DCG1000}\t\n')
    
    return test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000
    

