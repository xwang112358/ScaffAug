import torch
import torch.nn.functional as F
import os

atom_decoder = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']

def convert_graphs(molecule_list):
    """Convert the generated graphs to pyg graphs"""
    from torch_geometric.data import Data
    
    pyg_graphs = []
    for i, mol in enumerate(molecule_list):    
        atom_types, edge_types = mol
        x = F.one_hot(atom_types, num_classes=len(atom_decoder))
        # Add an extra column to x
        x = torch.cat([x, torch.zeros(x.size(0), 1)], dim=1)
        edge_index = torch.nonzero(edge_types > 0, as_tuple=True)
        edge_index = torch.stack(edge_index, dim=0)  # Convert to (2, num_edges) format
        # Get edge attributes and convert to 0-based indexing
        edge_attr = edge_types[edge_index[0], edge_index[1]] - 1  # Subtract 1 to convert to 0-based indexing
        edge_attr = F.one_hot(edge_attr.long(), num_classes=4)
        # Switch edge attributes for double bond (index 1) and aromatic bond (index 3)
        mask_double = torch.all(edge_attr == torch.tensor([0,1,0,0]), dim=1)
        mask_aromatic = torch.all(edge_attr == torch.tensor([0,0,0,1]), dim=1)
        
        # Create new tensor with swapped values
        edge_attr_new = edge_attr.clone()
        edge_attr_new[mask_double] = torch.tensor([0,0,0,1])
        edge_attr_new[mask_aromatic] = torch.tensor([0,1,0,0])
        edge_attr = edge_attr_new
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        pyg_graphs.append(data)
    return pyg_graphs


datasets = ['AID1798', 'AID463087', 'AID488997', 'AID2689', 'AID485290']
split_schemes = ['random_cv1', 'random_cv2', 'random_cv3', 'random_cv4', 'random_cv5',
                 'scaffold_seed1', 'scaffold_seed2', 'scaffold_seed3', 'scaffold_seed4', 'scaffold_seed5']

for dataset in datasets:
    for split_scheme in split_schemes:
        print(f"Processing {dataset} with {split_scheme}")
        try:
            dataset_graphs = torch.load(f'./augment_datasets/{dataset}_{split_scheme}_0.1_generated_graphs.pt')
            pyg_graphs = convert_graphs(dataset_graphs)
            print(f"Number of graphs: {len(pyg_graphs)}")
            torch.save(pyg_graphs, f'./augment_pyg_datasets/{dataset}_{split_scheme}_0.1_generated_pyg_graphs.pt')
        except Exception as e:
            print(f"Error processing {dataset}_{split_scheme}: {str(e)}")


#         break


# dataset = torch.load('./augment_datasets/AID1798_random_cv1_0.1_generated_graphs.pt')
# pyg_graphs = convert_graphs(dataset)
# print(len(pyg_graphs))
# print(pyg_graphs[0])
# torch.save(pyg_graphs, './augment_pyg_datasets/AID1798_random_cv1_0.1_generated_pyg_graphs.pt')