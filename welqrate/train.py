
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


def train(model, dataset, config, device, train_eval=False):
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
    
    loss_fn = BCEWithLogitsLoss()
    
    # load dataset info
    dataset_name = dataset.name
    root = dataset.root
    mol_repr = dataset.mol_repr
    
    # create loader
    split_dict = dataset.get_idx_split(split_scheme)
    train_loader = get_train_loader(dataset[split_dict['train']], batch_size, num_workers, seed)
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
    
    # timestamp = datetime.now().strftime("%m-%d-%H-%M")
    base_path = f'./results/{dataset_name}/{split_scheme}/{model_name}'
    model_save_path = os.path.join(base_path, f'{model_name}.pt')
    log_save_path = os.path.join(base_path, f'train.log')
    metrics_save_path = os.path.join(base_path, f'test_results.txt')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    
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
    test_logAUC, test_EF, test_DCG, test_BEDROC = get_test_metrics(model, test_loader, device, 
                                                                   save_per_molecule_pred=True,
                                                                   save_path=base_path)
    print(f'{model_name} at epoch {best_epoch} test logAUC: {test_logAUC:.4f} test EF: {test_EF:.4f} test DCG: {test_DCG:.4f} test BEDROC: {test_BEDROC:.4f}')
    with open(metrics_save_path, 'w+') as result_file:
        result_file.write(f'logAUC={test_logAUC}\tEF={test_EF}\tDCG={test_DCG}\tBEDROC={test_BEDROC}\t\n')
    
    
    

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



# def train_class(model, loader, optimizer, scheduler, device, loss_fn):
#     model.train()
#     loss_list = []

#     for i, batch in enumerate(tqdm(loader, miniters=100)):
#         batch = batch.to(device)

#         # Check for isolated nodes
#         num_nodes = batch.num_nodes
#         connection_counts = torch.zeros(num_nodes, dtype=torch.int32, device=device)
#         ones = torch.ones_like(batch.edge_index[0], dtype=torch.int32)

#         connection_counts = scatter_add(ones, batch.edge_index[0], dim=0, dim_size=num_nodes)
#         connection_counts += scatter_add(ones, batch.edge_index[1], dim=0, dim_size=num_nodes)

#         # Add self-loops to isolated nodes
#         isolated_nodes = torch.where(connection_counts == 0)[0]
#         if len(isolated_nodes) > 0:
#             self_loops = torch.stack([isolated_nodes, isolated_nodes], dim=0)
#             batch.edge_index = torch.cat([batch.edge_index, self_loops], dim=1)

#             # Update node features if necessary (e.g., for attention mechanisms)
#             if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
#                 # Create self-loop edge attributes (you may need to adjust this based on your edge attributes)
#                 self_loop_attr = torch.zeros(len(isolated_nodes), batch.edge_attr.size(1), device=device)
#                 batch.edge_attr = torch.cat([batch.edge_attr, self_loop_attr], dim=0)


#         y_pred = model(batch)
#         loss = loss_fn(y_pred.view(-1), batch.y.view(-1).float())
        
#         loss_list.append(loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#     loss = np.mean(loss_list) 
#     return loss


# def train_reg(model, loader, optimizer, scheduler, device, loss_fn):
    
#     model.train()
#     loss_list = []

#     for i, batch in enumerate(tqdm(loader, miniters=100)):
#         batch.to(device)
#         # assert batch.edge_index.max() < batch.x.size(0), f"Edge index {batch.edge_index.max()} exceeds number of nodes"
#         y_pred = model(batch)
        
#         loss = loss_fn(y_pred.view(-1), batch.activity_value.view(-1))
            
#         loss_list.append(loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#     loss = np.mean(loss_list)
#     return loss