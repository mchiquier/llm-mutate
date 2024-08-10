from torch.nn import functional as F
from PIL import Image
import torchvision.transforms as transforms
import clip
import torch
import pdb

def scoring_function(new_fn,img_batch):

    scores_pos, scores_mixed = new_fn(img_batch)

    return scores_pos, scores_mixed

class ImageBatchJoint:

    def __init__(self,model, img_size, vlm_scoring) -> None:
        self.vlm_scoring = vlm_scoring
        self.device = 'cuda:5'
        self.resize = transforms.Resize(img_size, interpolation=Image.BICUBIC)
        self.append_class = False
        self.model = model 
        self.temp = 0.07
    
    def reinit_images(self,images,classidx):
        encodings=F.normalize(self.model.encode_image(images.to(self.device)))
        
        self.img_encoding_train = encodings
        self.classidx_train = classidx

    def reinit_images_test(self,images,classidx):
        self.img_encoding_test = F.normalize(self.model.encode_image(images.to(self.device)))
        self.classidx_test = classidx
 
    def to_train(self):
        self.img_encoding = self.img_encoding_train
        self.classidx = self.classidx_train
        
    def to_test(self):
        self.img_encoding = self.img_encoding_test
        self.classidx = self.classidx_test
        
    def encode(self,attribute): 
        if self.vlm_scoring=="clip": 
            tokens = clip.tokenize(attribute).to(self.device)
            return F.normalize(self.model.encode_text(tokens))   
        else:
            text_encode = model.encode_text(attribute)
            return text_encode
    
    def score(self, classifiers):
        #print(classifiers.shape, self.img_encoding.shape)
        clsscores = (self.img_encoding @ classifiers)/self.temp
        cross_entropy_loss = torch.nn.CrossEntropyLoss()

        return 0.0, clsscores

class ImageBatch:

    def __init__(self,  model, scientific_names, img_size, scoring, pretrain) -> None:

        self.vlm_scoring = scoring
        self.scientific_names = scientific_names
        self.device = 'cpu'
        self.resize = transforms.Resize(img_size, interpolation=Image.BICUBIC)
        self.model = model 
        self.pretrain = pretrain
        self.img_size = img_size
        self.temp = 0.07
    
    def reinit_images_pretrain(self,pos_images, neg_images, train=True):
        if train:
            self.img_encoding_pos_train = F.normalize(self.model.encode_image(pos_images.to(self.device)))
            self.img_encoding_neg_train = F.normalize(self.model.encode_image(neg_images.to(self.device)))
        else:
            self.img_encoding_pos_test = F.normalize(self.model.encode_image(pos_images.to(self.device)))
            self.img_encoding_neg_test = F.normalize(self.model.encode_image(neg_images.to(self.device)))

    def reinit_images_jointtrain(self,images, classidx, train=True):
        if train:
            self.img_encoding_train = F.normalize(self.model.encode_image(images.to(self.device)))
            self.classidx_train = classidx
        else:
            self.img_encoding_test = F.normalize(self.model.encode_image(images.to(self.device)))
            self.classidx_test = classidx
        
    def to_train(self):
        if self.pretrain:
            self.img_encoding_pos = self.img_encoding_pos_train
            self.img_encoding_neg = self.img_encoding_neg_train
        else:
            self.img_encoding = self.img_encoding_train
            self.classidx = self.classidx_train

    def to_test(self):
        if self.pretrain:
            self.img_encoding_pos = self.img_encoding_pos_test
            self.img_encoding_neg = self.img_encoding_neg_test
        else:
            self.img_encoding = self.img_encoding_test
            self.classidx = self.classidx_test

    def encode(self,attribute): 
        if self.vlm_scoring=="clip": 
            tokens = clip.tokenize(attribute)
            return F.normalize(self.model.encode_text(tokens))   
        else:
            text_encode = self.model.encode_text(attribute)
            return text_encode
    
    def score(self, attr_encodings):
        if self.pretrain:
            with torch.no_grad(): 
                total = attr_encodings.T 
               
                output_pos =self.img_encoding_pos @ total 
                output_neg = self.img_encoding_neg @ total 

                return output_pos, output_neg
        else:
            
            with torch.no_grad():
                clsscores = (self.img_encoding @ attr_encodings)/self.temp
                cross_entropy_loss = torch.nn.CrossEntropyLoss()
                #pdb.set_trace()
                cross_entropy_score = cross_entropy_loss(torch.mean(clsscores,dim=2).T, self.classidx).item()
        
                #return accuracy
                return -cross_entropy_score, clsscores

