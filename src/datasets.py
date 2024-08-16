from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from os.path import join
import json
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from scipy.spatial import distance_matrix
import csv
import random
import pdb
import logging

path = '/proj/vondrick2/utkarsh/datasets/iNat2021/'

class PretrainDataset(Dataset):
    def __init__(self, root, pos_class, split='train', transform=None):
        self.pos_class = pos_class
        self.mode = split
        self.transform = transform
        self.root = root

        self.pos_images = []
        self.neg_images = []
        self.neg_labels = []

        # Set up logging
        logging.basicConfig(filename='dataset_errors.log', level=logging.ERROR)

        self.load_dataset()

    def load_dataset(self):
        dataset_path = os.path.join(self.root, self.mode)
        #pdb.set_trace()
        for i,class_name in enumerate(os.listdir(dataset_path)[:5]):
            #pdb.set_trace()
            print(i,5)
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                pdb.set_trace()
                continue
            
            label = int(i == self.pos_class)
            #pdb.set_trace()
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    
                    if label == 1:
                        self.pos_images.append((img, label))
                    else:
                        self.neg_images.append((img, label))
                        #self.neg_labels.append(int(class_name))
                except Exception as e:
                    logging.error(f"Error loading image {img_path}: {str(e)}")
    def __len__(self):
        return len(self.pos_images)

    
    def __getitem__(self, idx):
        pos_img, pos_label = self.pos_images[idx]
        neg_idx = random.randint(0, len(self.neg_images)-1)
        neg_img, neg_label = self.neg_images[neg_idx]
        
        return pos_img, neg_img

class INatDataset(Dataset):
    def __init__(self, root_dir, split, main_class, rest_classes, transform=None):
        self.root_dir = root_dir
        self.imgdir = join(root_dir, split)
        self.metafile = join(root_dir, split+'.json')
        self.transform = transform
        self.rest_classes = rest_classes
        self.main_class = main_class

        with open(self.metafile) as ifd:
            self.metadata = json.load(ifd)

        self.images_rest = [tmp for tmp in self.metadata['images'] if tmp['file_name'].split("/")[1] in self.rest_classes]
        new_order = np.arange(0,len(self.images_rest)).tolist()
        random.shuffle(new_order)
        
        self.images_rest = [self.images_rest[x] for x in new_order]
      
        #self.classindmap_rest = [self.classindmap_rest[x] for x in new_order]

        self.images_main = [tmp for tmp in self.metadata['images'] if tmp['file_name'].split("/")[1] in self.main_class]
        
    def __len__(self):
        return len(self.images_main)

    def __getitem__(self, idx):


        img_main = self.images_main[idx]
        img_rest = self.images_rest[idx]

        image_main = Image.open(join(self.root_dir, img_main['file_name']))
        image_rest = Image.open(join(self.root_dir, img_rest['file_name']))
        if self.transform is not None:
            image_main = self.transform(image_main)
            image_rest = self.transform(image_rest)
        return image_main, image_rest


class INatDatasetJoint(Dataset):
    def __init__(self, root_dir, cindset, split='train_mini', transform=None):
        self.root_dir = root_dir
        self.imgdir = join(root_dir, split)
        self.metafile = join(root_dir, split+'.json')
        self.transform = transform

        with open(self.metafile) as ifd:
            self.metadata = json.load(ifd)
        
        
        #cindset = set([3788,3789,3792,3793,3794,403,3127,5328,1978,6344])#set(self.classinds)
        #cindset = set([3788,3789,3792,3793,3794][:numclasses])
        #cindset= set([5438,5439,5440,5441,5442,5443])
        

        self.images = {tmp['id']: tmp for tmp in self.metadata['images']}
        self.temp = [tmp['class'] for tmp in self.metadata['categories']]
        
        self.classes = [tmp['common_name'] for tmp in self.metadata['categories'] if tmp['id'] in cindset]
        self.scientific_name = [tmp['name'] for tmp in self.metadata['categories']  if tmp['id'] in cindset]
        combined_name = set(['{} ({})'.format(sn, cn) for sn, cn in zip(self.scientific_name, self.classes)])
        self.classinds = [tmp['id'] for tmp in self.metadata['categories'] if tmp['id'] in cindset]
        self.annotation = [tmp for tmp in self.metadata['annotations'] if tmp["category_id"] in cindset]
        imgset = set([tmp["image_id"] for tmp in self.annotation])
        self.images = {k:v for k,v in self.images.items() if k in imgset}
        with open(join(self.root_dir, 'cbd_descriptors.json')) as ifd:
            self.cbd_descriptors = json.load(ifd)
        self.cbd_descriptors = {k:v for k,v in self.cbd_descriptors.items() if k in combined_name}
        self.classindmap = {i:ind for ind, i in enumerate(self.classinds)}
        self.common_name = {i:name for i, name in enumerate(self.classes)}
        print(self.classindmap, self.common_name)
        assert len(self.annotation)==len(self.images)
        assert len(self.cbd_descriptors)==len(self.classes)
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anno = self.annotation[idx]
        img = self.images[anno["image_id"]]
        label = self.classindmap[anno["category_id"]]
        image = Image.open(join(self.root_dir, img['file_name']))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_desc_byid(self, id):
        name = self.scientific_name[id]
        common_name = self.classes[id]
        key = '{} ({})'.format(name, common_name)
        return self.cbd_descriptors[key]


class INatDatasetJointCoarse(Dataset):
    def __init__(self, root_dir, split='train_mini', transform=None):
        self.root_dir = root_dir
        self.imgdir = join(root_dir, split)
        self.metafile = join(root_dir, split+'.json')
        self.transform = transform

        self.classindmap = {7709: 0,
        7710:0,
        7711:0,
        7712:0,
        7713:0,
        6297:1,
        6298:1,
        6299:1,
        6300:1,
        6301:1,
        6371:2,
        6372:2,
        6373:2,
        6374:2,
        6375:2,
        2843:3,
        2844:3,
        2845:3,
        2846:3,
        2847:3,
        5438:4,
        5439:4,
        5440:4,
        5441:4,
        5442:4,
        5443:4
        }

        cindset = list(self.classindmap.keys())

        with open(self.metafile) as ifd:
            self.metadata = json.load(ifd)
        
        self.images = {tmp['id']: tmp for tmp in self.metadata['images']}
        self.temp = [tmp['class'] for tmp in self.metadata['categories']]
        
        self.classes = [tmp['common_name'] for tmp in self.metadata['categories'] if tmp['id'] in cindset]
        self.scientific_name = [tmp['name'] for tmp in self.metadata['categories']  if tmp['id'] in cindset]
        combined_name = set(['{} ({})'.format(sn, cn) for sn, cn in zip(self.scientific_name, self.classes)])
        self.classinds = [tmp['id'] for tmp in self.metadata['categories'] if tmp['id'] in cindset]
        print(self.classinds)
        self.annotation = [tmp for tmp in self.metadata['annotations'] if tmp["category_id"] in cindset]
        imgset = set([tmp["image_id"] for tmp in self.annotation])
        self.images = {k:v for k,v in self.images.items() if k in imgset}
        with open(join(self.root_dir, 'cbd_descriptors.json')) as ifd:
            self.cbd_descriptors = json.load(ifd)
        self.cbd_descriptors = {k:v for k,v in self.cbd_descriptors.items() if k in combined_name}

        
        #self.classindmap = {i:ind for ind, i in enumerate(self.classinds)}
        self.common_name = {i:name for i, name in enumerate(self.classes)}
        print(self.classindmap) #self.common_name)
        assert len(self.annotation)==len(self.images)
        assert len(self.cbd_descriptors)==len(self.classes)
        

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        anno = self.annotation[idx]
        img = self.images[anno["image_id"]]
        label = self.classindmap[anno["category_id"]]
        image = Image.open(join(self.root_dir, img['file_name']))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_desc_byid(self, id):
        name = self.scientific_name[id]
        common_name = self.classes[id]
        key = '{} ({})'.format(name, common_name)
        return self.cbd_descriptors[key]
