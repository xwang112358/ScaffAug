from welqrate.dataset import WelQrateDataset

datasets = ['AID1798', 'AID463087', 'AID488997', 'AID2689', 'AID485290']

for dataset_name in datasets:
    dataset = WelQrateDataset(dataset_name=dataset_name, root='./welqrate_datasets', mol_repr='2dmol')

    print(f"Dataset {dataset_name} loaded with {len(dataset)} molecules")