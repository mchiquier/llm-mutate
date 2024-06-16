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
from inference_utils import *
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
    
    with open('inaturalist_species.json', 'r') as f:
        families = json.load(f)
    synset_ids = families[config.synset]["ids"]
    inat_dataloader = load_datasets_jointtrain(config.dataset_path, synset_ids, config.batch_size)
    inatdl = inat_dataloader['test']

    device="cuda:5"

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False) 
    imgbatch = ImageBatchJoint(model,224,'clip')
    thedescs_discovered = families[config.synset]["1prompt_desc"]
    # thedescs_discovered=[
    #     ['rocks that contain hornblende or amphibolite', 'rocks with a sandy texture and a reddish tint'],
    #     ['dainty, white flowers', 'flap-like bracts', 'stalked, gland-dotted leaves', 'furry, velvety stems', 'honey-scented Informally'],
    #     [' large clusters of small, fragrant flowers', ' glossy, dark green, hairy leaves', 'twigs with clusters of small, oval-shaped leaves', ' pair of occasionally coloured, petal-like leaves with red veins', ' slender, striated stems with thorns', ' many-stemmed, shrubby tree with smooth, gray bark'],
    #     ['twigs covered in hoar frost', 'hairy crown', 'spiders caught in a spider web', 'tendrils', 'interlocking branches and twigs', 'wispy, feathery foliage'],
    #     ['orthoclase', 'striated', 'biotite gneiss', ' rhombic', 'rapakivi', 'elephant hide'],
    #     ['cracks in a rock', 'normally found in warm areas', 'columnar basalt', 'unweathered rock surfaces', 'a fresh vegetable', 'worn, rounded stones']
    # ]

    for x, y in inatdl:
        print("here")
        x = x.to(device)
        y = y.to(device)
        imgbatch.reinit_images_test(x,y)
        #pdb.set_trace()

    evaluate_llm_mutate(synset_ids, thedescs_discovered, imgbatch)

    ### ZERO SHOT DESCRIPTORS ###

    # thedescs = [inatds.get_desc_byid(i) for i in range(len(inatds.classes))]
    # calculate_classonly(imgbatch, inatds, inatdl)
    # calculate_cbd_results(inatds, thedescs, imagenet_templates, imgbatch, inatdl)

    # thedescscopy = thedescs

    #### GREEDY REMOVE ON DISCOVERED ######
    # thedescs=[
    #     ['Red-winged Blackbird', ' Blue Jay', ' American Robin ', 'yellow or orange bill', 'Black-capped Chickadee', 'red patch on its wing'],
    #           ['a black and white striped belly', 'a long, pointed bill', 'black and white striped wings', 'gray and white feathers', 'a silvery-gray bird', 'a bird feeder'],
    #            ['a robin'],
    #             ['female house sparrow', 'house sparrow', 'juvenile house sparrow', 'juvenile Tree Sparrow', 'Tree Sparrow', 'Eurasian Tree Sparrow'],
    #              ['a juvenile  female western tanager', 'a gray-and-black bird', 'a sharp-shinned hawk', 'a western flycatcher', 'a willow flycatcher', 'an olive-sided flycatcher']]

    # thedescs=[
    #     ['splendid fairy-bluebird', 'blue-crowned motmot', 'baltimore oriole', 'a pair of birds perched together', 'a black stripe above the eye and a crest', 'golden-winged warbler'],
    #     ['black and white striped feathers', 'black and white striped tail', 'black and white striped wings', 'black with white patches'],
    #     ['a creek running', 'a large tree', 'a brown bird', 'a red-breasted nuthatch', 'gray-crowned nuthatch', 'a sunlit forest'],
    #     ['juvenile house sparrow', 'adult purple finch', 'female purple finch', 'male purple finch', 'juvenile tree sparrow', 'adult house sparrow'],
    #     ['a blue-gray bird with a rust-colored patch on its breast', 'a warbling vireo with a white eye ring', 'a male western tanager perched on a branch', 'a bird with black-and-white stripes on its head, a gray body, and a black tail with white outer feathers', 'a western tanager with a reddish face, a black throat, and a yellow head', 'gray-and-white']
    # ]

    #berries
    # thedescs_discovered = [ [' prickly acacia tree', 'thorns，drooping branches', 'thorns， erect branches', 'evergreen', ' yellow flowers'],
    # ['chickweed', 'cuckoo flower', 'sweet woodruff', 'lambs lettuce', 'daisy fleabane', 'whitlow grass'],
    # ['chartreuse leaves with a hint of pink', 'twisted, curved branches', 'heart-shaped leaves with a silvery sheen', 'fragrant, trumpet-shaped flowers', 'a tree with a unique, spiral- patterned bark', 'long, thin leaves with a velvety texture'],
    # ['ozzie (Australian Parrot)', ' Eastern red cedar', ' weeping habit', ' drooping branches', ' pungent heartwood', ' insect resistant timber'],
    # ['Fragaria borealis', 'wild strawberry', 'red clover', 'sweet woodruff', 'Fragaria vesca', 'dwarf veronica']]

    # thedescs_discovered = [['Helenium amarantoides', 'Rudbeckia hirta', 'Rudbeckia tricolor', 'Cyperus blumei ssp. borealis', 'Silphium terebinthinaceum', 'Helianthus debilis'],
    #             ['ornamental grasses with dynamic movement', 'low-lying shrubs with a golden hue', 'plumes of pampas grass with seasonal color changes', 'ornamental grasses with unique textures', 'tall ornamental grasses with feathery plumes', 'dried grasses with interesting seed heads'],
    #             ['sharp, jagged edges on the teeth', 'greenish-white flowers', 'evergreen foliage', 'saltwater or seawater', 'white flowers', 'a common weed with small white flowers'],
    #             ['dried flowers', 'pressed flowers', 'preserved leaves', 'petals with lines', 'smooth petals', 'long stamens'],
    #             ['dense spikes of small blue flowers and pale yellow-green, lanceolate leaves', 'broad green leaves and dense panicles of pinkish flowers', 'aromatic foliage and prominent whitish veins', 'hairy stems and dense, terminal spikes of purple flowers', 'dense whorls of green leaves and spreading growth habit', 'angular branches and shiny, dark green leaves']
    #             ]

    # thedescs_discovered=[['a blacktip reef shark swimming through a group of blue tang', 'a school of butterflyfish swimming through a channel of sea fans', 'a yellowtail damselfish and a blue-and-yellow damselfish swimming through a maze', 'a clownfish peeking out of an anemone', 'a foxface rabbitfish swimming through a group of blue-and-yellow damselfish', 'a trumpetfish swimming through a group of moorish idols'],
    #         ['a fish with three or four horizontal stripes on a grey background', 'a fish with vertical zebra stripes having white and dark grey colors'],
    #         ['a school of fish swimming together near a coral reef', 'an underwater scene of a large, prehistoric-looking fish swimming alone', 'a group of dolphins swimming together near a sunken ship', 'an underwater scene of a school of snakes swimming together in the open water', 'a group of flamingos swimming together in shallow water', 'a large group of jellyfish swimming together near the surface'],
    #         ['a vibrant, rainbow-colored parrotfish swimming through a surreal, dreamlike underwater scene', 'a blue tang with a shimmering, iridescent pattern', 'a green-and-orange-colored parrotfish with a distorted, wavy appearance', 'a yellow-and-blue-colored parrotfish with a colorful, sparkling background', 'a purple-and-blue-colored parrotfish with a glowing, neon-like aura', 'a neon-colored parrotfish with an electric, glowing effect'],
    #             ['a swordtail blenny', 'a very small shrimp', 'a periodic Sch Coris', 'typically brown with yellow fins', 'a small striped blenny', 'a spotted lionfish']]


    # thedescs_discovered=[['a blacktip reef shark swimming through a group of blue tang', 'a school of butterflyfish swimming through a channel of sea fans', 'a yellowtail damselfish and a blue-and-yellow damselfish swimming through a maze', 'a clownfish peeking out of an anemone', 'a foxface rabbitfish swimming through a group of blue-and-yellow damselfish', 'a trumpetfish swimming through a group of moorish idols'],
    #          ['a fish with three or four horizontal stripes on a grey background', 'a fish with vertical zebra stripes having white and dark grey colors'],
    #          ['a fish swimming together near a coral reef', 'an underwater scene of a large, prehistoric-looking fish swimming alone', 'a dolphin swimming near a sunken ship', 'an underwater scene of a snake swimming in the open water', 'a flamingo swimming in shallow water', 'a jellyfish swimming near the surface'],
    #            ['a vibrant, rainbow-colored parrotfish swimming through a surreal, dreamlike underwater scene', 'a blue tang with a shimmering, iridescent pattern', 'a green-and-orange-colored parrotfish with a distorted, wavy appearance', 'a yellow-and-blue-colored parrotfish with a colorful, sparkling background', 'a purple-and-blue-colored parrotfish with a glowing, neon-like aura', 'a neon-colored parrotfish with an electric, glowing effect'],
    #             ['a swordtail blenny', 'a very small shrimp', 'a periodic Sch Coris', 'typically brown with yellow fins', 'a small striped blenny', 'a spotted lionfish']]


    

    # thedescs_discovered = [['a serene countryside scene with fields of ripe wheat and a distant, hazy horizon', 'fields of golden wheat blowing in the breeze', 'a bale of hay in a peaceful meadow', 'fields of tall, vibrant grasses and colorful wildflowers', 'fields of bright, yellow sunflowers', 'a field of green, young wheat with a large, old tree in the background'],
    #             ['desert lavender', 'creosote bush', 'arizona barrel cactus', 'sand sagebrush', 'spiny hedgehog cactus', 'teddy bear cholla'],
    #             ['birch-like bark', 'spiky, dark green leaves', 'diamond-shaped leaves', 'cool, humid ravines', ' cliff-dwelling', ' edible nuts', 'long, spreading branches', 'clusters of greenish flowers', ' twigs with reddish, swollen nodes'],
    #             ['rows of long, thin, parallel strips', 'uneven upper edge', 'a ring or coil with small, green leaves', 'a long, thin, flexible rod with uneven, roughened surfaces', 'a piece of jewelry with a sharp, pointed spine', 'typically made of leather with a long, thin strap'],
    #             ['  smooth to finely pubescent', ' compound palmate', 'leaves Miami near lakeshores', 'may have small hairs on leaves', 'may form large colonies', 'important to waterfowl']]

    

    # list_of_scores=[]
    # for mode in ['sname','cname','both']:

    #     if mode=="cname":
    #         classes = inatds.classes
    #     elif mode=="sname":
    #         classes = inatds.scientific_name
    #     elif mode=="both":
    #         classes = ['{} ({})'.format(name, common_name) for name, common_name in zip(inatds.classes, inatds.scientific_name)]

    #     clsfeats = []

    #     with torch.no_grad():
    #         for i in range((len(classes)-1)//batch_size+1):
    #             cnames = classes[i*batch_size:min(i*batch_size+batch_size, len(classes))]
    #             tex_feats = imgbatch.encode(cnames)#tex_enc(**tokenizer(cnames, padding=True, return_tensors="pt").to(device)).text_embeds
    #             clsfeats.append(tex_feats)
    #     clsfeats = torch.cat(clsfeats, dim=0).T
    #     #print(clsfeats.shape)
    #     clsmatrix = np.zeros((len(inatds.classes), len(inatds.classes)))
    #     topnmatrix = np.zeros((len(inatds.classes), 2))
    #     topn = 5

    #     with torch.no_grad():
    #         for x, y in inatdl:
    #             x = x.to(device)
    #             y = y.to(device)
    #             #print(x.shape)
    #             imgbatch.reinit_images(x,y)#img_enc(x).image_embeds
    #             imgbatch.to_test()
    #             #feats = F.normalize(feats, dim=1)
    #             cross_score, clsscores = imgbatch.score(clsfeats)
    #             pred = torch.argmax(clsscores, dim=1)
    #             topk = torch.topk(clsscores, topn, dim=1)[1]
    #             intopk = torch.sum(topk==y.unsqueeze(dim=-1), dim=1)
    #             for t, p, top in zip(y, pred, intopk):
    #                 clsmatrix[t, p]+=1
    #                 if top>0.5:
    #                     topnmatrix[t, 0]+=1
    #                 else:
    #                     topnmatrix[t, 1]+=1
    #             acc = np.trace(clsmatrix)/np.sum(clsmatrix)
    #             macc = np.nanmean(np.diag(clsmatrix)/np.sum(clsmatrix, axis=1))
    #             topkmacc = np.nanmean((topnmatrix[:, 0])/np.sum(topnmatrix, axis=1))
    #             topkacc = np.sum(topnmatrix[:, 0])/np.sum(topnmatrix)
    #         print(mode, np.round(acc*100, 2), np.round(macc*100, 2), np.round(topkacc*100, 2), np.round(topkmacc*100, 2))
    #         list_of_scores.append(np.round(acc*100, 2))   
    #         #pdb.set_trace()



    # for mode in ["sname+template","cname+template","both+template"]:
    #     if mode == "cname+template":
    #         thedescscopy = [t.format(inatds.classes[i]) for t in imagenet_templates]
    #     elif mode == "sname+template":
    #         thedescscopy = [t.format(inatds.scientific_name[i]) for t in imagenet_templates]
    #     else:
    #         thedescscopy = [t.format('{} ({})'.format(inatds.classes[i], inatds.scientific_name[i])) for t in imagenet_templates]
    #     desfeats = []
    #     with torch.no_grad():
    #         for i in range(len(inatds.classes)):
                
                
    #             descs = thedescscopy[i]#load_gpt_descriptions('', thedescs[i])
    #                 #print(descs)
    #             # elif mode =="both":
    #             #     descs = inatds.get_desc_byid(i)
    #             #     descs = descs+['{} ({})'.format(inatds.classes[i], inatds.scientific_name[i])]
    #             # print(descs)
    #             #pdb.set_trace()
    #             desfeatcur = imgbatch.encode(descs).T
    #         # tex_feats = model.encode_text(clip.tokenize(descs).to(device))
    #             #tex_feats = tex_enc(**tokenizer(descs, padding=True, return_tensors="pt").to(device)).text_embeds
    #             desfeats.append(desfeatcur)
    #             # print(i, desfeats[-1].shape)
    #         clsmatrix = np.zeros((len(inatds.classes), len(inatds.classes)))
    #         topnmatrix = np.zeros((len(inatds.classes), 2))
    #         topn = 3
    #         with torch.no_grad():
    #             for x, y in inatdl:
    #                 #print("here")
    #                 x = x.to(device)
    #                 y = y.to(device)
    #                 imgbatch.reinit_images(x,y)#img_enc(x).image_embeds
    #                 imgbatch.to_test()
    #                 clsscores = []
    #                 for i,desfeat in enumerate(desfeats):
    #                     #pdb.set_trace()
    #                     #print(i,mode)
    #                     _, scores = imgbatch.score(desfeat)

    #                     # print(scores.shape)
    #                     clsscores.append(torch.mean(scores,dim=1))
    #                     #pdb.set_trace()
    #                 clsscores = torch.stack(clsscores).T
    #                 pred = torch.argmax(clsscores, dim=1)
    #                 topk = torch.topk(clsscores, topn, dim=1)[1]
    #                 intopk = torch.sum(topk==y.unsqueeze(dim=-1), dim=1)
    #                 for t, p, top in zip(y, pred, intopk):
    #                     clsmatrix[t, p]+=1
    #                     if top>0.5:
    #                         topnmatrix[t, 0]+=1
    #                     else:
    #                         topnmatrix[t, 1]+=1
    #                 acc = np.trace(clsmatrix)/np.sum(clsmatrix)
    #                 macc = np.nanmean(np.diag(clsmatrix)/np.sum(clsmatrix, axis=1))
    #                 topkmacc = np.nanmean((topnmatrix[:, 0])/np.sum(topnmatrix, axis=1))
    #                 topkacc = np.sum(topnmatrix[:, 0])/np.sum(topnmatrix)
    #             print(mode, np.round(acc*100, 2), np.round(macc*100, 2), np.round(topkacc*100, 2), np.round(topkmacc*100, 2))
    #             # break




    # thedescs = [inatds.get_desc_byid(i) for i in range(len(inatds.classes))]

    # for mode in ["cbd","cbd+template"]:
    #     desfeats = []
    #     with torch.no_grad():
    #         for i in range(len(inatds.classes)):
            
    #             if mode== "cbd+template":
    #                 descs_iter = thedescs[i]#load_gpt_descriptions('', thedescs[i])
    #                 descs = []  
    #                 for desc in descs_iter:
    #                     descs.extend([t.format(desc) for t in imagenet_templates])

    #             else:
    #                 descs = thedescs[i]
    #             desfeatcur = imgbatch.encode(descs).T
    #             desfeats.append(desfeatcur)


    #         clsmatrix = np.zeros((len(inatds.classes), len(inatds.classes)))
    #         topnmatrix = np.zeros((len(inatds.classes), 2))
    #         topn = 2
    #         with torch.no_grad():
    #             #img_enc(x).image_embeds
    #             imgbatch.to_test()
    #             clsscores = []
    #             for i,desfeat in enumerate(desfeats):

    #                 _, scores = imgbatch.score(desfeat)

    #                 clsscores.append(torch.mean(scores,dim=1))
    #                 #pdb.set_trace()
    #             clsscores = torch.stack(clsscores).T
    #             pred = torch.argmax(clsscores, dim=1)
    #             topk = torch.topk(clsscores, topn, dim=1)[1]
    #             y=imgbatch.classidx
    #             intopk = torch.sum(topk==y.unsqueeze(dim=-1), dim=1)
    #             for t, p, top in zip(y, pred, intopk):
    #                 clsmatrix[t, p]+=1
    #                 if top>0.5:
    #                     topnmatrix[t, 0]+=1
    #                 else:
    #                     topnmatrix[t, 1]+=1
    #             acc = np.trace(clsmatrix)/np.sum(clsmatrix)
    #             macc = np.nanmean(np.diag(clsmatrix)/np.sum(clsmatrix, axis=1))
    #             topkmacc = np.nanmean((topnmatrix[:, 0])/np.sum(topnmatrix, axis=1))
    #             topkacc = np.sum(topnmatrix[:, 0])/np.sum(topnmatrix)
    #         print("zero shot cbd", mode, np.round(acc*100, 2), np.round(macc*100, 2), np.round(topkacc*100, 2), np.round(topkmacc*100, 2))


    # for classtype in ["sname","cname","both"]:
    #     if classtype=="sname":
    #         thedescs = [load_gpt_descriptions(inatds.scientific_name[i],thedescs[i]) for i in range(len(thedescs))]

    #     elif classtype=="cname":
    #         thedescs = [load_gpt_descriptions(inatds.classes[i],thedescs[i]) for i in range(len(thedescs))]

    #     else:
    #         thedescs = [load_gpt_descriptions('{} ({})'.format(inatds.classes[i], inatds.scientific_name[i]), thedescs[i]) for i in range(len(thedescs))]

    #     for mode in ["cbd","cbd+template"]:
    #         desfeats = []
    #         with torch.no_grad():
    #             for i in range(len(inatds.classes)):
                
    #                 if mode== "cbd+template":
    #                     descs_iter = thedescs[i]#load_gpt_descriptions('', thedescs[i])
    #                     descs = []  
    #                     for desc in descs_iter:
    #                         descs.extend([t.format(desc) for t in imagenet_templates])
    #                 else:
    #                     descs = thedescs[i]

    #                 desfeatcur = imgbatch.encode(descs).T
    #                 desfeats.append(desfeatcur)


    #             clsmatrix = np.zeros((len(inatds.classes), len(inatds.classes)))
    #             topnmatrix = np.zeros((len(inatds.classes), 2))
    #             topn = 2
    #             with torch.no_grad():
    #                 #img_enc(x).image_embeds
    #                 imgbatch.to_test()
    #                 clsscores = []
    #                 for i,desfeat in enumerate(desfeats):

    #                     _, scores = imgbatch.score(desfeat)

    #                     clsscores.append(torch.mean(scores,dim=1))
    #                     #pdb.set_trace()
    #                 clsscores = torch.stack(clsscores).T
    #                 pred = torch.argmax(clsscores, dim=1)
    #                 topk = torch.topk(clsscores, topn, dim=1)[1]
    #                 y=imgbatch.classidx
    #                 intopk = torch.sum(topk==y.unsqueeze(dim=-1), dim=1)
    #                 for t, p, top in zip(y, pred, intopk):
    #                     clsmatrix[t, p]+=1
    #                     if top>0.5:
    #                         topnmatrix[t, 0]+=1
    #                     else:
    #                         topnmatrix[t, 1]+=1
    #                 acc = np.trace(clsmatrix)/np.sum(clsmatrix)
    #                 macc = np.nanmean(np.diag(clsmatrix)/np.sum(clsmatrix, axis=1))
    #                 topkmacc = np.nanmean((topnmatrix[:, 0])/np.sum(topnmatrix, axis=1))
    #                 topkacc = np.sum(topnmatrix[:, 0])/np.sum(topnmatrix)
    #             print("zero shot class + cbd", classtype, mode, np.round(acc*100, 2), np.round(macc*100, 2), np.round(topkacc*100, 2), np.round(topkmacc*100, 2))



    