from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
#from inference_utils import *
import pdb
import clip
from utilities import load_data, imagenet_templates
from config import Config
from data_loader import load_datasets_jointtrain, make_descriptor_sentence, load_gpt_descriptions
from image_batch import ImageBatchJoint
from torchvision import datasets, transforms


def evaluate_llm_mutate(synset_ids, thedescs_discovered, imgbatch):
    for mode in ["cbd","cbd+template"]:
        desfeats = []
        with torch.no_grad():
            for i in range(len(synset_ids)):
                if mode== "cbd+template":
                    descs_iter = thedescs_discovered[i]
                    descs = []  
                    for desc in descs_iter:
                        descs.extend([t.format(desc) for t in imagenet_templates])
                else:
                    descs= thedescs_discovered[i] 

                desfeatcur = imgbatch.encode(descs).T
                desfeats.append(desfeatcur)


            clsmatrix = np.zeros((len(synset_ids), len(synset_ids)))
            topnmatrix = np.zeros((len(synset_ids), 2))
            topn = 2
            with torch.no_grad():
                imgbatch.to_test()
                clsscores = []

                for i,desfeat in enumerate(desfeats):
                    _, scores = imgbatch.score(desfeat)
                    clsscores.append(torch.mean(scores,dim=1))

                clsscores = torch.stack(clsscores).T
                pred = torch.argmax(clsscores, dim=1)
                topk = torch.topk(clsscores, topn, dim=1)[1]
                y=imgbatch.classidx
                intopk = torch.sum(topk==y.unsqueeze(dim=-1), dim=1)

                for t, p, top in zip(y, pred, intopk):
                    clsmatrix[t, p]+=1
                    if top>0.5:
                        topnmatrix[t, 0]+=1
                    else:
                        topnmatrix[t, 1]+=1

                acc = np.trace(clsmatrix)/np.sum(clsmatrix)
                macc = np.nanmean(np.diag(clsmatrix)/np.sum(clsmatrix, axis=1))
                topkmacc = np.nanmean((topnmatrix[:, 0])/np.sum(topnmatrix, axis=1))
                topkacc = np.sum(topnmatrix[:, 0])/np.sum(topnmatrix)
            print("accuracies (acc, macc, topkmacc, topkacc): ", mode, np.round(acc*100, 2), np.round(macc*100, 2), np.round(topkacc*100, 2), np.round(topkmacc*100, 2))

if __name__ == "__main__":
    config = Config()
    
    with open('files/inaturalist_species.json', 'r') as f:
        families = json.load(f)
    synset_ids = families[config.synset]["ids"]

    if config.dataset_name=='iNaturalist':

        inat_dataloader = load_datasets_jointtrain(config.dataset_path, synset_ids, config.batch_size, config.experiment)
        inatdl = inat_dataloader['test']

        if config.experiment != "ours":
            descs = inat_dataloader['descs']
        else:
            descs = families[config.synset]["desc"]

    else:
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
        kiki_dataset = datasets.ImageFolder(root=config.dataset_path+"val", transform=transform)
        inatdl = DataLoader(kiki_dataset, batch_size=len(kiki_dataset), shuffle=False, drop_last=False, pin_memory=True, num_workers=32)
        json_file_path = config.dataset_path + 'kiki_bouba.json'
        classes = list(kiki_dataset.class_to_idx.keys())

        # Open the JSON file and load the data into a dictionary
        with open(json_file_path) as json_file:
            cbd_data = json.load(json_file)

        if 'cbd' in config.experiment:
            descs_only = [value for key,value in cbd_data.items()]
            descs = [load_gpt_descriptions(classes[i], descs_only[i]) for i in range(len(descs_only))]
        elif 'zero_shot' in config.experiment:
            descs = [value for key,value in cbd_data.items()]
        elif 'clip' in config.experiment:
            descs = [[class_name] for class_name in classes]
        elif config.experiment == "ours":
            if config.dataset_name=='KikiBouba_v1':
                descs=[[' twisting trunk', ' long, sloping branches', ' dense, dark green foliage', '  sharp, thorn-like leaf tips', ' coppery-red or dark purple bark', ' tiny, cream-colored flowers'],
                ['minimalist Isometric cityscape surreal futuristic tech Triforce lego'],
                ['a blue-skinned, 8-armed deity with a glowing third eye, standing on a lotus flower, surrounded by a halo of loving-kindness, under a rainbow, with a chorus of singing cherubs and a harp in the background, while holding a crystal orb and spreading blessings to all beings'],
                ['sections of shimmering, glowing citrus fruits', 'sections of iridescent, shimmering peacock feathers', 'a series of colorful, wavy shapes resembling a coral reef', 'sections of luminous, glowing mushrooms', 'a delicate, lacy pattern made of glistening, dew-covered spider webs', 'sections of soft, fluffy clouds'],
                ['two small, circular shapes with thin black outlines', 'F-15 written in red on the tail']]
            elif config.dataset_name=='KikiBouba_v2':
                descs=[['a glow-in-the-dark bowling ball', 'a mini bowling ball', 'a bowling ball with different colored holes', 'a bowling ball with a face', 'a bowling ball with a variety of different textures', 'a bowling ball that is surrounded by 3 ghostly bowling balls'],
                        ['flying wing aircraft', 'delta wing aircraft', 'swing-wing aircraft', 'variable geometry wing aircraft', 'rotorcraft', 'tiltrotor aircraft'],
                        ['two long, curved tusks', 'a fluffy tail', 'a bone, decorative comb', 'a sharp-tipped, flexible hunting knife', 'a tough, animal hide pouch', 'a beaded, leather cord'],
                        ['origami', 'a palace', 'a sorting system', 'snow-covered', 'grassy area', 'revolving circular platform'],
                        ['groovy colorful background', 'psychedelic colors', 'holographic', 'shimmering', 'iridescent colors', 'colorful patterns']        ]
            else:
                descs = None 
                print("Error with config.dataset_name value")
                
        else:
            descs = None 
            print("Error with config.experiment value")

        synset_ids = [i for i in range(len(descs))]

    if descs:

        device="cuda:5"

        model, preprocess = clip.load("ViT-B/32", device=device, jit=False) 
        imgbatch = ImageBatchJoint(model,224,'clip')


        for x, y in inatdl:
            x = x.to(device)
            y = y.to(device)
            imgbatch.reinit_images_test(x,y)

        evaluate_llm_mutate(synset_ids, descs, imgbatch)
