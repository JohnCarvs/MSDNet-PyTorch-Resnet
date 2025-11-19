import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os


def get_imagenetc_dataloader(args, split='test'):
    """
    Generic function to load ImageNet-C datasets from any split directory.
    
    Returns:
        List of (corruption, severity, loader) tuples
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Store list of (corruption, severity, loader) tuples
    loader_list = []
    data_root = os.path.join(args.data_root, split)
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Directory not found: {data_root}")
    
    corruption_types = sorted([d for d in os.listdir(data_root) 
                              if os.path.isdir(os.path.join(data_root, d))])
    
    for corruption in corruption_types:
        for severity in range(1, 6):
            folder = os.path.join(data_root, corruption, str(severity))
            if os.path.exists(folder):
                dataset = datasets.ImageFolder(folder, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                ]))
                loader_single = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=True
                )
                loader_list.append((corruption, severity, loader_single))
    
    return loader_list


def get_val_dataloader(args, split='val_split'):
    """
    Loads ImageNet-C validation split.
    """
    return get_imagenetc_dataloader(args, split=split)


def get_test_dataloader(args, split='test_split'):
    """
    Loads ImageNet-C test split.
    """
    return get_imagenetc_dataloader(args, split=split)
