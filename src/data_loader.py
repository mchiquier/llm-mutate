from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets import INatDataset, INatDatasetJoint
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

def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"

def load_gpt_descriptions(classname, descs):
    moddescs = [] 
    for desc in descs:
        moddescs.append(classname+', '+make_descriptor_sentence(desc))
    return moddescs

def load_datasets_jointtrain(dataset_path, cindset, batch_size, experiment, transform=None):
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

        if experiment=="zero_shot":
            descs = [load_gpt_descriptions('', train_dataset.get_desc_byid(i)) for i in range(len(train_dataset.classes))]
        elif experiment=="cbd_scientific":
            descs = [load_gpt_descriptions(train_dataset.scientific_name[i], train_dataset.get_desc_byid(i)) for i in range(len(train_dataset.classes))]
        elif experiment=="cbd_common": 
            descs = [load_gpt_descriptions(train_dataset.common_name[i], train_dataset.get_desc_byid(i)) for i in range(len(train_dataset.classes))]
        elif experiment=="clip_scientific":
            descs = [train_dataset.scientific_name[i] for i in range(len(train_dataset.classes))]
        elif experiment=="clip_common":
            descs = [train_dataset.common_name[i] for i in range(len(train_dataset.classes))]
        else:
            descs = None

        # Creating data loaders for both datasets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)

        return {'train': train_loader, 'test': test_loader, 'descs': descs}

    except Exception as e:
        print(f"Failed to load datasets: {e}")
        return None