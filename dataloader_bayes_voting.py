import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os


def get_val_dataloader(args):
    """Loads ImageNet validation using datasets.ImageNet"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    # Use pytorch official ImageNet dataset class
    val_set = datasets.ImageNet(
        root=args.data_root,  # dir that contains val/
        split='val',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )
    
    # Create DataLoader with standard settings for inference
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True
    )
    
    return val_loader

def get_test_dataloader(args):
    """Loads ImageNet-C (corrupted) datasets"""
    test_loader = None

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    
    # store list of (corruption, severity, loader) tuples
    test_sets = []
    test_root = os.path.join(args.data_root, 'test')
    
    corruption_types = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
    #corruption_types = ['defocus_blur']
    for corruption in corruption_types:
        for severity in range(1, 6):
            folder = os.path.join(test_root, corruption, str(severity))
            if os.path.exists(folder):
                test_set = datasets.ImageFolder(folder, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                ]))
                test_loader_single = torch.utils.data.DataLoader(
                    test_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=True
                )
                test_sets.append((corruption, severity, test_loader_single))
    # Skip use_valid logic for ImageNet-C
    return test_sets
