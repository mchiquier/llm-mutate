from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from inat_dataloader_refactor import INatDataset, INatDatasetJoint
import pdb

def load_datasets_pretrain(dataset_path, main_class, rest_classes, batch_size, transform=None):
    """
    Loads datasets and returns PyTorch data loaders for training and testing.

    Args:
    dataset_path (str): Path to the dataset directory.
    batch_size (int): Number of items in each batch of data.
    transform (dict): Transformations to apply to the dataset images; expects 'train' and 'test' keys.

    Returns:
    dict: A dictionary containing 'train' and 'test' DataLoader objects.
    """
    if transform is None:
        # Default transformations if none provided
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    try:
        # Loading the image datasets from the specified path
        train_dataset = INatDataset(dataset_path, split='train', main_class=main_class, rest_classes=rest_classes, transform=transform)
        test_dataset =  INatDataset(dataset_path, split='val', main_class=main_class, rest_classes=rest_classes, transform=transform)

        # Creating data loaders for both datasets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        #pdb.set_trace()

        return {'train': train_loader, 'test': test_loader}
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        return None

def load_datasets_jointtrain(dataset_path, cindset, batch_size, transform=None):
    """
    Loads datasets and returns PyTorch data loaders for training and testing.

    Args:
    dataset_path (str): Path to the dataset directory.
    batch_size (int): Number of items in each batch of data.
    transform (dict): Transformations to apply to the dataset images; expects 'train' and 'test' keys.

    Returns:
    dict: A dictionary containing 'train' and 'test' DataLoader objects.
    """
    if transform is None:
        # Default transformations if none provided
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    try:
        # Loading the image datasets from the specified path
        train_dataset = INatDatasetJoint(dataset_path, split='train', cindset=cindset, transform=transform)
        test_dataset =  INatDatasetJoint(dataset_path, split='val',cindset=cindset, transform=transform)

        # Creating data loaders for both datasets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)
        #pdb.set_trace()
        return {'train': train_loader, 'test': test_loader}
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        return None