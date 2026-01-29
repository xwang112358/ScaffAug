# ScaffAug
Scaffold-Aware Generative Augmentation and Reranking for Enhanced Virtual Screening


## Environment Installation

```bash
conda create -c conda-forge -n scaffaug rdkit=2023.03.2 python=3.9
```

```bash
conda activate scaffaug
```

```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

```bash
pip install -r requirements.txt
```

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

```bash
pip install -e .
```

Navigate to the ./src/analysis/orca directory and compile orca.cpp:

```bash
g++ -O2 -std=c++11 -o orca orca.cpp
```

## Usage Guide for finetune_deepgin_scaffaug_st.py

The `finetune_deepgin_scaffaug_st.py` script is used for fine-tuning DeepGIN models with scaffold-aware augmentation using self-training and pseudo-labeling techniques. This script performs hyperparameter optimization using Optuna and evaluates the best model across multiple random seeds.

### Unzip augmentation dataset 

unzip augment_valid_pyg_graphs_labels.zip


### Command Line Arguments

The script requires three mandatory arguments:

- `--dataset`: Dataset name (e.g., 'AID1798', 'AID485290', 'AID2689')
- `--split`: Data split scheme (e.g., 'random_cv1', 'scaffold_seed1')  
- `--sampling_method`: Sampling method for augmented data (e.g., 'sabs')

### Basic Usage

```bash
python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split random_cv1 --sampling_method sabs
```

### Example Commands

For different datasets and splits:

```bash
# Random cross-validation splits
python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split random_cv1 --sampling_method sabs
python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split random_cv2 --sampling_method sabs

# Scaffold-based splits  
python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split scaffold_seed1 --sampling_method sabs
python finetune_deepgin_scaffaug_st.py --dataset AID485290 --split scaffold_seed2 --sampling_method sabs
```

### Hyperparameter Optimization

The script uses Optuna for hyperparameter optimization with the following search space:

- **Embedding dimension**: [64, 128, 256]
- **Number of layers**: 3-5 
- **Dropout ratio**: 0.1-0.3
- **Graph pooling**: Fixed to "sum"
- **Peak learning rate**: 1e-4 to 1e-2 (log scale)
- **Confidence threshold**: [0.75, 0.8, 0.85, 0.9]
- **Pseudo label frequency**: 3-10 epochs
- **Start epoch for pseudo-labeling**: 20-30

The optimization runs 16 trials by default and maximizes the test BEDROC score.

### Output Files

The script generates several output files in the `scaffaug_deepgin_st_{sampling_method}_results/` directory:

1. **Hyperparameter tuning results**: 
   ```
   deepgin_scaffaug_st_finetuning_{dataset}_{split}_{timestamp}.csv
   ```

2. **Final results across seeds**:
   ```  
   deepgin_scaffaug_st_final_{dataset}_{split}_{timestamp}.csv
   ```

3. **Summary statistics**:
   ```
   deepgin_scaffaug_st_summary_stats_{dataset}_{split}_{timestamp}.csv
   ```


