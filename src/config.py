import os
import torchvision.transforms as transforms

class Config:
    def __init__(self):
        # Model related configurations
        self.scoring = 'clip'  # could be 'clip', 'xvlm', etc.
        self.max_iter = 600
        self.experiment = 'ours' #choose from ['zero_shot', 'clip_scientific', 'clip_common', 'cbd_scientific', 'cbd_common', 'ours']
        self.classifiers_initialized = 1000 
        self.number_of_classifiers_in_prompt = 10
        self.batch_size = 2000
        self.llm_type = 'llama'  # could be 'gpt-3', 'codex', 'llama'
        self.replacement = False
        self.random_scores_order = False
        self.synset = 'Lichen' 
        self.dataset_name = 'Art' #'iNaturalist', 'KikiBouba_v2', 'KikiBouba_v1'
        self.objective = 'binarymean'
        self.append_class = False
        self.do_pretraining = False
        self.pretraining_path = None #'results/binaryclassifier_iNaturalist_Lichen_10prompt/llama_clip_classapp_False_objective_binarymean_bs_2000/'
        #self.synset_ids = set([5438,5439,5440,5441,5442,5443])
        self.per_index_gen = 10

        # API credentials
        self.api_key = os.getenv('OPENAI_API_KEY', 'default_api_key')

        # Dataset specific configurations
        self.dataset_path = '/proj/vondrick4/mia/painters_v0/'#'/proj/vondrick2/utkarsh/datasets/iNat2021/'
        #self.dataset_path = '/proj/vondrick4/mia/kiki_bouba_v2_split/'
         
        # Image transformation settings
        self.image_size = 224  # default size, can be overridden by model settings
        self.normalize_mean = [0.48145466, 0.4578275, 0.40821073]
        self.normalize_std = [0.26862954, 0.26130258, 0.27577711]
        
        # Environment configurations
        self.device = 'cpu'  # Use 'cuda' for GPU

        # Configuration for hyperparameters if different from default
        self.hparams = {
            'image_size': 224,
            'device': self.device,
            'model_size': "ViT-B/32" if self.scoring == 'clip' else None
        }

        # Transformation operations, possibly reused in data loaders or preprocessing steps
        self.transformation = {
            'train': self.get_transform(),
            'test': self.get_transform(is_train=False)
        }

    def get_transform(self, is_train=True):
        """
        Generates a torchvision transform pipeline for image preprocessing.
        """
        if is_train:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])