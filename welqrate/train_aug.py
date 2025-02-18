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
from torch_geometric.data import Data
from torch_geometric.data import Dataset

def get_train_loss(model, loader, optimizer, scheduler, device, loss_fn):
    
    model.train()
    loss_list = []

    for i, batch in enumerate(tqdm(loader, miniters=100)):
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


class CombinedDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list
        
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]

def train(model, orig_dataset, aug_dataset, config, device, train_eval=False):

    # load train info
    batch_size = int(config['TRAIN']['batch_size'])
    num_epochs = int(config['TRAIN']['num_epochs'])
    num_workers = int(config['GENERAL']['num_workers'])
    seed = int(config['GENERAL']['seed'])
    weight_decay = float(config['TRAIN']['weight_decay'])
    early_stopping_limit = int(config['TRAIN']['early_stop'])
    split_scheme = config['DATA']['split_scheme']
    
    loss_fn = BCEWithLogitsLoss()
    
    # load dataset info
    dataset_name = orig_dataset.name

    print(aug_dataset[0])
    # create loader
    split_dict = orig_dataset.get_idx_split(split_scheme)
    train_list = []
    for graph in tqdm(orig_dataset, desc="Processing original dataset"):
        train_list.append(Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=graph.y))
    train_list.extend(aug_dataset)

    # train_dataset = CombinedDataset(train_list)
    # create train, valid, test loaders
    train_loader = get_train_loader(train_list, batch_size, num_workers, seed)
    valid_loader = get_valid_loader(orig_dataset[split_dict['valid']], batch_size, num_workers, seed)
    test_loader = get_test_loader(orig_dataset[split_dict['test']], batch_size, num_workers, seed) 

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
    base_path = f'./results_aug/{dataset_name}/{split_scheme}/{model_name}0'
    version = 0
    while os.path.exists(base_path):
        version += 1
        base_path = f'./results_aug/{dataset_name}/{split_scheme}/{model_name}{version}'
    
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
            
            train_loss = get_train_loss(model, train_loader, optimizer, scheduler, device, loss_fn)
            
            lr = get_lr(optimizer)
            
            if train_eval:
                train_logAUC, train_EF, train_DCG, train_BEDROC = get_test_metrics(model, train_loader, device)
                print(f'current_epoch={epoch} train_loss={train_loss:.4f} lr={lr}')
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
    test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = get_test_metrics(model, test_loader, device, 
                                                                   save_per_molecule_pred=True,
                                                                   save_path=base_path, extra_metrics=True)
    print(f'{model_name} at epoch {best_epoch} test logAUC: {test_logAUC:.4f} test EF: {test_EF100:.4f} test DCG: {test_DCG100:.4f} test BEDROC: {test_BEDROC:.4f}')
    with open(metrics_save_path, 'w+') as result_file:
        result_file.write(f'logAUC={test_logAUC}\tEF100={test_EF100}\tDCG100={test_DCG100}\tBEDROC={test_BEDROC}\tEF500={test_EF500}\tEF1000={test_EF1000}\tDCG500={test_DCG500}\tDCG1000={test_DCG1000}\n')
    
    return test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000
    

def get_test_metrics(model, loader, device, type = 'test', 
                     save_per_molecule_pred=False, save_path=None, extra_metrics=False):
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
    EF100 = cal_EF(all_true_y, all_pred_y, 100)
    DCG100 = cal_DCG(all_true_y, all_pred_y, 100)
    BEDROC = cal_BEDROC_score(all_true_y, all_pred_y)

    if extra_metrics:
        EF500 = cal_EF(all_true_y, all_pred_y, 500)
        EF1000 = cal_EF(all_true_y, all_pred_y, 1000)
        DCG500 = cal_DCG(all_true_y, all_pred_y, 500)
        DCG1000 = cal_DCG(all_true_y, all_pred_y, 1000)
        return logAUC, EF100, DCG100, BEDROC, EF500, EF1000, DCG500, DCG1000
    else:
        return logAUC, EF100, DCG100, BEDROC



