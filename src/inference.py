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
from data_loader import load_datasets_jointtrain
from image_batch import ImageBatchJoint



def evaluate_llm_mutate(synset_ids, thedescs_discovered, imgbatch):
    for mode in ["cbd","cbd+template"]:
        desfeats = []
        with torch.no_grad():
            for i in range(len(synset_ids)):
            
                if mode== "cbd+template":
                    descs_iter = thedescs_discovered[i]#load_gpt_descriptions('', thedescs[i])
                    descs = []  
                    for desc in descs_iter:
                        #print(desc)
                        descs.extend([t.format(desc) for t in imagenet_templates])

                else:
                    descs= thedescs_discovered[i] 
                desfeatcur = imgbatch.encode(descs).T
                desfeats.append(desfeatcur)


            clsmatrix = np.zeros((len(synset_ids), len(synset_ids)))
            topnmatrix = np.zeros((len(synset_ids), 2))
            topn = 2
            with torch.no_grad():
                #img_enc(x).image_embeds
                imgbatch.to_test()
                #pdb.set_trace()
                clsscores = []
                for i,desfeat in enumerate(desfeats):

                    _, scores = imgbatch.score(desfeat)

                    clsscores.append(torch.mean(scores,dim=1))
                    #pdb.set_trace()
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
            print("discovered attributes", mode, np.round(acc*100, 2), np.round(macc*100, 2), np.round(topkacc*100, 2), np.round(topkmacc*100, 2))

if __name__ == "__main__":
    config = Config()
    
    with open('files/inaturalist_species.json', 'r') as f:
        families = json.load(f)
    synset_ids = families[config.synset]["ids"]
    inat_dataloader = load_datasets_jointtrain(config.dataset_path, synset_ids, config.batch_size)
    inatdl = inat_dataloader['test']

    device="cuda:5"

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False) 
    imgbatch = ImageBatchJoint(model,224,'clip')
    thedescs_discovered = families[config.synset]["desc"]

    for x, y in inatdl:
        print("here")
        x = x.to(device)
        y = y.to(device)
        imgbatch.reinit_images_test(x,y)

    evaluate_llm_mutate(synset_ids, thedescs_discovered, imgbatch)
