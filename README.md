# ScaffAug
Scaffold-aware Augmentation for HTS Datasets via Diffusion Model


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
pip install -e .
```

Navigate to the ./src/analysis/orca directory and compile orca.cpp:

```bash
g++ -O2 -std=c++11 -o orca orca.cpp
```


