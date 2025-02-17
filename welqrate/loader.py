from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch

# need to change for regression task
def get_train_loader(train_dataset, batch_size, num_workers, seed):
    num_train_active = len(torch.nonzero(torch.tensor([data.y for data in train_dataset])))
    num_train_inactive = len(train_dataset) - num_train_active
    print(f'training # of molecules: {len(train_dataset)}, actives: {num_train_active}')

    train_sampler_weight = torch.tensor([(1. / num_train_inactive)
                                         if data.y == 0
                                         else (1. / num_train_active)
                                         for data in
                                         train_dataset])

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_sampler = WeightedRandomSampler(weights=train_sampler_weight,
                                          num_samples=len(
                                          train_sampler_weight),
                                          generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader



def get_test_loader(test_dataset, batch_size, num_workers, seed):
    num_test_active = len(torch.nonzero(torch.tensor([data.y for data in test_dataset])))
    print(f'test # of molecules: {len(test_dataset)}, actives: {num_test_active}')

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader

def get_valid_loader(valid_dataset, batch_size, num_workers, seed):
    num_valid_active = len(torch.nonzero(torch.tensor([data.y for data in valid_dataset])))
    print(f'validation # of molecules: {len(valid_dataset)}, actives: {num_valid_active}')

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return valid_loader

def get_self_train_loader(combined_dataset, batch_size, num_workers, seed):
    # Get original and augmented datasets from combined dataset
    orig_dataset = combined_dataset.orig_dataset
    aug_dataset = combined_dataset.aug_dataset
    
    # Calculate statistics for original dataset
    orig_y = torch.tensor([data.y for data in orig_dataset])
    num_orig_active = len(torch.nonzero(orig_y))
    num_orig_inactive = len(orig_dataset) - num_orig_active
    
    print("\nOriginal Dataset Statistics:")
    print(f"Total molecules: {len(orig_dataset)}")
    print(f"Active molecules: {num_orig_active}")
    print(f"Inactive molecules: {num_orig_inactive}")
    
    # Calculate statistics for augmented dataset (only confident predictions)
    permanent_indices = sorted(list(combined_dataset.permanent_confident_indices))
    if permanent_indices:
        aug_labels = aug_dataset.pseudo_labels[permanent_indices]
        num_aug_active = len(torch.nonzero(aug_labels))
        num_aug_inactive = len(permanent_indices) - num_aug_active
        
        print("\nConfident Augmented Dataset Statistics:")
        print(f"Total molecules: {len(permanent_indices)}")
        print(f"Active molecules: {num_aug_active}")
        print(f"Inactive molecules: {num_aug_inactive}")
        
        print("\nCombined Dataset Statistics:")
        total_active = num_orig_active + num_aug_active
        total_inactive = num_orig_inactive + num_aug_inactive
        print(f"Total molecules: {len(combined_dataset)}")
        print(f"Active molecules: {total_active}")
        print(f"Inactive molecules: {total_inactive}\n")
    else:
        print("\nNo confident predictions in augmented dataset yet\n")
        total_active = num_orig_active
        total_inactive = num_orig_inactive

    # Create weights for balanced sampling
    train_sampler_weight = []
    
    # Weights for original dataset
    for data in orig_dataset:
        weight = 1.0 / num_orig_active if data.y == 1 else 1.0 / num_orig_inactive
        train_sampler_weight.append(weight)
    
    # Weights for augmented dataset
    for idx in permanent_indices:
        label = aug_dataset.pseudo_labels[idx]
        weight = 1.0 / total_active if label == 1 else 1.0 / total_inactive
        train_sampler_weight.append(weight)
    
    train_sampler_weight = torch.tensor(train_sampler_weight)
    
    # Create sampler
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    train_sampler = WeightedRandomSampler(
        weights=train_sampler_weight,
        num_samples=len(train_sampler_weight),
        generator=generator
    )
    
    # Create loader
    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader