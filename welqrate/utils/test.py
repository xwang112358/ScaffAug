import os
import sys
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.abspath(os.getcwd()))
import torch
from torch_scatter import scatter_add
from welqrate.utils.rank_prediction import rank_prediction
from welqrate.utils.evaluation import calculate_logAUC, cal_EF, cal_DCG, cal_BEDROC_score, \
                       MAE, MSE, RMSE, R2

def get_test_metrics(model, loader, device, type = 'test', 
                     save_per_molecule_pred=False, 
                     save_path=None, extra_metrics=False):
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