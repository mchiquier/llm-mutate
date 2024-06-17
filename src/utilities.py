import clip
import torch
import pdb
import pathlib
from PIL import Image
import torchvision.transforms as transforms
import json
import random
import os
from tqdm import tqdm
import pdb
import numpy as np
import ast
import torch
import copy
import time
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.nn import functional as F
from openai import OpenAI
from joblib import Memory
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import aiohttp
import asyncio
import pickle
import concurrent.futures


from collections import defaultdict
import re
import csv
import pdb
from os.path import join
from llama import Llama
import json

import os

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]
# generator = Llama.build(
#     ckpt_dir="llama3/Meta-Llama-3-70B-Instruct",
#     tokenizer_path="llama3/Meta-Llama-3-70B-Instruct/tokenizer.model",
#     max_seq_len=1024,
#     max_batch_size=256,
# )
# message_list = ['What is the capital of France?',"What is the capital of England?"]

# print("here")

# results = generator.text_completion(
#                 message_list, #[message],
#                 max_gen_len=256,
#                 temperature=0.6,
#                 top_p=0.9,
#             )

# print(results[0]['generation'], results[1]['generation'])

cache = Memory('cache/', verbose=0)

nycdata_transforms = {
    "train": transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


def OpenSource(theprompt, index, client):
    try:
        #pdb.set_trace()
        completion = client.completions.create(model="meta-llama/Llama-2-70b-chat-hf",prompt=theprompt, max_tokens=200)
        
        return completion.choices[0].text, index
    except:
        return ''
    
def LlamaThree(theprompt, index, client):
    try:
        c = Client()
        return c(theprompt)
    except:
        return ''

def call_OpenSource(theprompt, index, client):
    return OpenSource(theprompt, index, client)

def call_llamathree(theprompt, index, client):
    return LlamaThree(theprompt, index, client)


def call_GPT4(theprompt, client):
    return GPT4(theprompt, client)



# def OpenSource(theprompt, client):
#     completion = client.completions.create(model="meta-llama/Llama-2-70b-chat-hf",
#                                       prompt=[theprompt]*20, max_tokens=200)
    
#     pdb.set_trace()
    
#     return completion.choices[0].text
OPENAI_API_KEY="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"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}
#@cache.cache
async def get_completion(client, content, session, semaphore):
    async with semaphore:
        await asyncio.sleep(1)

        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model="gpt-4-0125-preview"
        )
        try:
            result = chat_completion.choices[0].message.content
        except IndexError:
            print(chat_completion)
            result = "Error: Response format is not as expected"

        return result

async def get_completion_list(client, content_list, max_parallel_calls):
    semaphore = asyncio.Semaphore(value=max_parallel_calls)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = [get_completion(client, content, session, semaphore) for content in content_list]
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)

    return results



def GPT4(theprompt,client):
    #print("calling API")
    #pdb.set_trace()
    prompts=[theprompt]*20
    response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompts,
    max_tokens=100,
    ) 
    # chat_completion = client.chat.completions.create(

    # messages=[
    #     {
    #         "role": "user",
    #         "content": theprompt,
    #     },
    # ],
    # model="gpt-4-0125-preview",
    # )
    # content = chat_completion.choices[0].message.content
    stories = [""] * len(prompts)
    for choice in response.choices:
        stories[choice.index] =  choice.text



    return stories


def create_folder_and_log_exp_details(expname, llm, scoring,append_class,objective,class_string,  batch_size):
    prefix = 'results/' + expname + '/' + llm + "_" + scoring + "_classapp_" + str(append_class)  + "_objective_" + objective  + "_bs_" + str(batch_size) + "/"
    cur_folder = prefix + class_string.replace(" ", "_") 

    os.makedirs(cur_folder,exist_ok=True)
    iter=0
    while True:
        file=cur_folder + "/experiment_" + str(iter)
        if "experiment_" + str(iter) in os.listdir(cur_folder):
            iter=iter+1
        else:
            os.mkdir(file)
            break

    os.mkdir(file + "/scores_train/")
    os.mkdir(file + "/scores_test/")
    return file


def remove_list_parentheses(input_string):
    # Find the start and end indices of the list
    start_index = input_string.find("[")
    end_index = input_string.rfind("]")

    if start_index != -1 and end_index != -1:
        # Extract the list content
        list_content = input_string[start_index:end_index + 1]
        # Remove parentheses and their content from the list content
        list_content_cleaned = re.sub(r'\([^)]*\)', '', list_content)
        # Replace the original list content with the cleaned one
        output_string = input_string[:start_index] + list_content_cleaned + input_string[end_index + 1:]
        return output_string
    else:
        # If the list delimiters are not found, return the original string
        return input_string

def extract_attributes_from_string(s):
    pattern = r"\[([^\[\]]+)\]"
    match = re.search(pattern, s)
    if match:
        attributes_str = match.group(1)
        attributes = [attr.strip().strip("'") for attr in attributes_str.split(',')]
        return attributes
    return None


def extract_after_numbered_strings(input_string):
    # Use re.findall to find all occurrences of numbered strings followed by text
    matches = re.findall(r'\d+\.\s+(.*)', input_string)
    return matches

def truncate_list_in_function(input_string):
    # Find the start and end indices of the list inside image.score([...])
    match = re.search(r'image\.score\((\[[^\[\]]*\])\)', input_string)
    if match:
        list_string = match.group(1)
        elements = re.findall(r"'[^']*'", list_string)

        if len(elements) < 6:
            # Repeat the list until it reaches 6 elements
            repetitions = (6 // len(elements)) + 1
            repeated_list = elements * repetitions
            truncated_list = repeated_list[:6]
        else:
            # Truncate the list to the first 6 elements
            truncated_list = elements[:6]

        # Join the truncated elements back into a list string
        truncated_list_string = "[" + ", ".join(truncated_list) + "]"

        # Replace the original list string with the truncated one in the input_string
        output_string = re.sub(r'image\.score\(\[[^\[\]]*\]\)', f'image.score({truncated_list_string})', input_string)
        return output_string
    else:
        # If the list inside image.score([...]) is not found, return the original string
        return input_string
    

def has_repeats(lst):
    return len(lst) != len(set(lst))


def float_keys_hook(dct):
    return {float(key): value for key, value in dct.items()}


def extract_newfun_definition(input_string, function_name):
    # Define the pattern to match the function definition
    pattern = r"def\s+" + re.escape(function_name) + r"\(.*?\)\s*:\s*return\s+image\.score\(\[.*?\]\)"

    # Use regular expression to find the pattern in the input string
    match = re.search(pattern, input_string, re.DOTALL)

    if match:
        # Extract and return the matched substring
        return match.group()
    else:
        # Return None if no match is found
        return None

def escape_quotes(your_string):
    return your_string.replace('"', '\\"')

def extract_after_numbered_strings(input_string):
    # Use re.findall to find all occurrences of numbered strings followed by text
    matches = re.findall(r'\d+\.\s+(.*)', input_string)
    return matches

def extract_newfun_definition(input_string, function_name):
    # Define the pattern to match the function definition
    pattern = r"def\s+" + re.escape(function_name) + r"\(.*?\)\s*:\s*return\s+image\.score\(\[.*?\]\)"

    # Use regular expression to find the pattern in the input string
    match = re.search(pattern, input_string, re.DOTALL)

    if match:
        # Extract and return the matched substring
        return match.group()
    else:
        # Return None if no match is found
        return None



def make_descriptor_sentence(descriptor, key, append_class):
    if append_class:
        if descriptor.startswith('a') or descriptor.startswith('an'):
            return f"a {key} which is {descriptor}"
        elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
            return f"a {key} which {descriptor}"
        elif descriptor.startswith('used'):
            return f"a {key} which is {descriptor}"
        else:
            return f"a {key} which has {descriptor}"
    else:
        return descriptor

def transform_tensor(tensor):
    if tensor.size(0) < 6:
        # Repeat the first dimension to make the tensor (6, 512)
        repeated_tensor = tensor.repeat((6 // tensor.size(0)) + 1, 1)[:6]
        return repeated_tensor
    elif tensor.size(0) > 6:
        # Cut off the last dimensions to make the tensor (6, 512)
        cut_tensor = tensor[:6]
        return cut_tensor
    else:
        # The tensor is already of shape (6, 512)
        return tensor
    
def get_accuracy(rawscore,labels):
    _, predicted = torch.max(torch.mean(rawscore,dim=2), 0)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

def plot_patches(patches,size):

    reshaped_tensor = patches[0].permute((1,2,0,3,4)).reshape(-1,3,size,size)
    num=patches[0].shape[2]
    fig, axs = plt.subplots(num, num, figsize=(10, 10))
    for i in range(num):
        for j in range(num):
            axs[i, j].imshow(reshaped_tensor[i * num + j].permute(1, 2, 0))
            axs[i, j].axis('off')
    plt.savefig("patch_" + str(size) + ".png")
    #pdb.set_trace()
    #vutils.save_image(pos_images[0], "image_" + str(size) + ".png")
class GSVData(Dataset):
    def __init__(self, csvfile, imdir,  index, main_class,  split='train', transform=None):
        self.csvfile = csvfile
        self.imdir = imdir
        self.transform = transform
        self.index = index
        self.main_class = main_class
        self.split = split

        self.fnames_main = []
        self.labels_main = []

        self.fnames_other = []
        self.labels_other = []
        with open('output.txt', 'r') as f: 
            self.accepted_files = [line.strip() for line in f.readlines()]
        with open(csvfile) as ifd:
            reader = csv.reader(ifd)
            for i, row in enumerate(reader):
                if i==0 or row[0] not in self.accepted_files:
                    print(row[0])
                    continue
               
                if self.index==3 or self.index==4:
                    if int(row[index]) == main_class:
                        self.fnames_main.append(row[0])
                        self.labels_main.append(int(row[index]))
                    else:
                        self.fnames_other.append(row[0])
                        self.labels_other.append(int(row[index]))
                else:
                    if int(row[index]) == main_class:
                        self.fnames_main.append(row[0])
                        self.labels_main.append(float(row[index]))
                    else:
                        self.fnames_other.append(row[0])
                        self.labels_other.append(float(row[index]))
        assert len(self.fnames_main)==len(self.labels_main)
        if self.index==3 or self.index==4:
            self.nclasses = len(np.unique(self.labels_main))

        # managing split
        data_indices_by_class = defaultdict(list)
        for i, label in enumerate(self.labels_other):
            data_indices_by_class[label].append(i)
        
        self.train_indices = []
        self.val_indices = []
        split_ratio = 0.8
        np.random.seed(42)

        np.random.shuffle(self.labels_other)
        np.random.shuffle(self.fnames_other)
        
        class_indices = np.arange(0,len(self.labels_other)).tolist()
        np.random.shuffle(class_indices)
        split_index = int(split_ratio * len(class_indices))
        self.train_indices = np.arange(0,int(0.8*len(self.labels_main))).tolist() #[].extend(class_indices[:split_index])
        self.val_indices = np.arange(int(0.8*len(self.labels_main)), len(self.labels_main)).tolist() #.extend(class_indices[split_index:])

        # sanity check
        # print(np.unique([self.labels[tmp] for tmp in self.train_indices], return_counts=True))
        # print(np.unique([self.labels[tmp] for tmp in self.val_indices], return_counts=True))

    def __len__(self):
        if self.split=='train':
            return len(self.train_indices)
        elif self.split=='val':
            return len(self.val_indices)

    def __getitem__(self, index):
        if self.split=='train':
            index = self.train_indices[index]
        elif self.split=='val':
            index = self.val_indices[index]
        fname_main, label_main = self.fnames_main[index], self.labels_main[index]
        image_main = Image.open(join(self.imdir, fname_main))
        fname_other, label_other = self.fnames_other[index], self.labels_other[index]
        image_other = Image.open(join(self.imdir, fname_other))
        if self.transform:
            image_main = self.transform[self.split](image_main)
            image_other = self.transform[self.split](image_other)
        return image_main, image_other, image_other, label_other

        
class ImageNet(Dataset):
    def __init__(self, root_dir,class_name='class1', allowed_classes=None,train=True, transform=None,opposite_id_dict=None, loader=default_loader):


        #allowed_classes = list(opposite_id_dict.keys())

        self.class_imgs = [os.path.join(root_dir, class_name, img) for img in os.listdir(os.path.join(root_dir, class_name))]
        self.class_two_imgs = [os.path.join(root_dir, class_name, img) for img in os.listdir(os.path.join(root_dir, class_name))]# class_ in os.listdir(root_dir) if class_ != class_name for img in os.listdir(os.path.join(root_dir, class_))]
        self.notclass_imgs = [os.path.join(root_dir, class_, img) for class_ in allowed_classes for img in os.listdir(os.path.join(root_dir, class_))]
        random.shuffle(self.notclass_imgs)
        self.train = train
        self.transform = transform
        self.opposite_id_dict=opposite_id_dict
        self.loader = loader

        #split the data into train and test
        
        self.truelength =min(len(self.class_imgs), len(self.class_two_imgs))

    def __len__(self):
        
        return self.truelength

    def __getitem__(self, idx):

        idx = idx%self.truelength
        classidx = self.opposite_id_dict[self.notclass_imgs[idx].split("/")[7]]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        random.shuffle(self.notclass_imgs)
        class1_image = Image.open(self.class_imgs[idx]).convert("RGB")
        class2_image = Image.open(self.class_two_imgs[idx]).convert("RGB")
        class3_image = Image.open(self.notclass_imgs[idx]).convert("RGB")


        if self.transform:
            class1_image = self.transform(class1_image)
            class2_image = self.transform(class2_image)
            class3_image = self.transform(class3_image)

        return class1_image, class2_image, class3_image,classidx
    



def extract_from_quotes(input_string):
    return re.findall(r"'(.*?)'", input_string)


def extract_word_from_function_definition(string):
    pattern = r"image\.score\(\[(.*?)\]\)"
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return None



def process_list_of_tensors(input_list, target_length=6):
    """
    Process a list of tensors by repeating or cutting off tensors to reach the target length.

    Args:
    - input_list (list): A list of PyTorch tensors.
    - target_length (int): The target length for the list.

    Returns:
    - processed_list (list): The processed list of tensors.
    """

    # Check if the input list is shorter than the target length
    if len(input_list) < target_length:
        # Repeat the tensors in the input list until it reaches the target length
        repeated_list = input_list * (target_length // len(input_list)) + input_list[:target_length % len(input_list)]
        return repeated_list
    else:
        # Cut off the last tensors from the input list to match the target length
        processed_list = input_list[:target_length]
        return processed_list

def get_meanlist_from_scores(scores_pos, scores_mixed):
    per_word_mean = torch.mean(scores_pos,dim=0) - torch.mean(scores_mixed,dim=0)
    per_word_mean = per_word_mean.tolist()
    #total_mean = torch.mean(torch.max(scores_pos,dim=1)[0]) - torch.mean(torch.max(scores_mixed,dim=1)[0])
    total_mean = torch.mean(scores_pos) - torch.mean(scores_mixed)
    per_word_mean.append(total_mean.item())
    return per_word_mean

# def evaluate_generation_and_log_jointtrain(jointtrain_file, num_classes, information, classifier_bank, img_batch, dict_of_generated_programs, iteration):
#     [scores_picked, programs_encoded_picked, scores_raw_picked, scores_raw, scores, programs_string, programs_encoded, gen_program_encoded, gen_raw_scores, gen_beforeandafter] = information 
#     best_prompt_score_index = torch.argmax(torch.tensor(scores_picked))
#     best_prompt_program=programs_encoded_picked[best_prompt_score_index]
#     #pdb.set_trace()

#     scores_crossentropy_bestprompt_train, scores_raw_bestprompt_train = scores_picked[best_prompt_score_index], scores_raw_picked[best_prompt_score_index]

#     best_llm_index = torch.argmax(scores).item()
#     best_llm_program_string = programs_string[best_llm_index]
#     best_llm_program_encoded = programs_encoded[best_llm_index]
#     #pdb.set_trace()

#     scores_crossentropy_bestbank_train, scores_raw_bestbank_train = scores[best_llm_index].item(),scores_raw[best_llm_index]
#     accuracy_train_best = get_accuracy(scores_raw_bestbank_train,img_batch.classidx)

#     print("best bank score: " , scores_crossentropy_bestbank_train)

#     img_batch.to_test()
#     #pdb.set_trace()

#     scores_crossentropy_bestprompt_test, scores_raw_bestprompt_test = img_batch.score(best_prompt_program)
#     scores_crossentropy_bestbank_test, scores_raw_bestbank_test = img_batch.score(best_llm_program_encoded)
#     accuracy_test_best = get_accuracy(scores_raw_bestbank_test,img_batch.classidx)

#     #pdb.set_trace()
#     np.save(jointtrain_file + "/scores_train/" + str(iteration) + "_raw_bestbank.npy",scores_raw_bestbank_train.detach().numpy())
#     np.save(jointtrain_file + "/scores_train/" + str(iteration) + "_raw_bestprompt.npy",scores_raw_bestprompt_train.detach().numpy())
#     np.save(jointtrain_file + "/scores_test/" + str(iteration)  + "_raw_bestbank.npy",scores_raw_bestbank_test.detach().numpy())
#     np.save(jointtrain_file + "/scores_test/" + str(iteration)  + "_raw_bestprompt.npy",scores_raw_bestprompt_test.detach().numpy())
#     #_trace()

#     with open(jointtrain_file + '/best_program.txt', 'a') as b:     
#         with open(jointtrain_file + '/scores_best.txt', 'a') as s:
#             s.write("iteration: " + str(iteration) + " train cross-entropy: " + str(scores_crossentropy_bestbank_train) + " test cross-entropy: " + str(scores_crossentropy_bestbank_test) 
#                     + " train accuracy: " + str(accuracy_train_best) + " test accuracy: " + str(accuracy_test_best) + "\n")
#             perclass = "".join(["class " + img_batch.scientific_names[i] + ": " + str(best_llm_program_string[i]) + "\n" for i in range(num_classes)])
#             b.write("iteration:  " + str(iteration) + " best program: \n" + perclass)

#     #pdb.set_trace()
#     if not os.path.isdir(jointtrain_file + "/program_bank"): 
#         os.mkdir(jointtrain_file + "/program_bank")
#     with open(jointtrain_file + "/program_bank/iter_" + str(iteration) +  ".pkl", 'wb') as f: pickle.dump(classifier_bank, f)
#     #pdb.set_trace()
#     pdb.set_trace()
#     accuracy_train_gen = get_accuracy(gen_raw_scores,img_batch.classidx)
#     np.save(jointtrain_file+ "/scores_train/" + str(iteration) +"_" +str(index)+ "_index.npy",img_batch.classidx.detach().numpy())
#     np.save(jointtrain_file + "/scores_train/" + str(iteration) +"_" +str(index) + "_raw_gen.npy",gen_raw_scores.detach().numpy())
#     pdb.set_trace()
#     img_batch.to_test()

#     pdb.set_trace()
#     gen_cross_entropy_score_test, gen_raw_scores_test = img_batch.score(gen_program_encoded)
#     accuracy_test_gen = get_accuracy(gen_raw_scores_test,img_batch.classidx)
#     pdb.set_trace()
#     np.save(jointtrain_file + "/scores_test/" + str(iteration) +"_" +str(index)+ "_index.npy",img_batch.classidx.detach().numpy())
#     np.save(jointtrain_file + "/scores_test/" + str(iteration) +"_" +str(index)+ "_raw_gen.npy",gen_raw_scores_test.detach().numpy())
    
#     with open(jointtrain_file +'/generations.txt', 'a') as c:
#         with open(jointtrain_file + '/scores_gen.txt', 'a') as q:
#             c.write("iteration: " + str(iteration) + " index: " + train_dataset.common_name[index] + " generation: " + gen_beforeandafter + "\n")
#             q.write("iteration: " + str(iteration) + " index: " + train_dataset.common_name[index] + " train cross-entropy: " + str(gen_cross_entropy_score) + " test cross-entropy: " + str(gen_cross_entropy_score_test) + " train accuracy: " + str(accuracy_train_gen) + " test accuracy: " + str(accuracy_test_gen) + "\n")

def best(img_batch, result, index, programs_encoded_picked_cur_preclone, programs_strings_picked_cur_preclone):

        programs_encoded_picked_cur = programs_encoded_picked_cur_preclone.clone()
        programs_strings_picked_cur = copy.deepcopy(programs_strings_picked_cur_preclone)
        list_of_scores = []
        list_of_rawscores = []
        list_of_programs_encoded =[]
        list_of_programs_str =[]
        list_of_beforeandafter = []
        completion = extract_newfun_definition(result, "newfun")
        img_batch.to_train()

        if completion: 
            try: 
                #encoding the attributes
                programs_str_gen = extract_attributes_from_string(completion)
                programs_enc_gen = transform_tensor(img_batch.encode(programs_str_gen)).T
                
                for p in range(len(programs_encoded_picked_cur)):
                    programs_encoded_picked_cur[p][index] = programs_enc_gen
                    beforeandafter = "before: " + str(programs_strings_picked_cur[p][index]) + "\n after: " + str(programs_str_gen)
                    programs_strings_picked_cur[p][index] = programs_str_gen
                    score, rawscores = img_batch.score(programs_encoded_picked_cur[p])
                    list_of_scores.append(score)
                    list_of_rawscores.append(rawscores)
                    list_of_programs_encoded.append(programs_encoded_picked_cur[p])
                    list_of_programs_str.append(programs_strings_picked_cur[p])
                    list_of_beforeandafter.append(beforeandafter)

                best_index = list_of_scores.index(max(list_of_scores))
                newscore = list_of_scores[best_index]
                best_program_encoded = list_of_programs_encoded[best_index]
                best_program_string= list_of_programs_str[best_index]
                newrawscore = list_of_rawscores[best_index]
                beforeandafter = list_of_beforeandafter[best_index]
                return [newscore, newrawscore], best_program_encoded, best_program_string, beforeandafter, index
            
            except Exception as error:
                    print("error here", error)
                    return None, None,None, None, None
        else:
            return None,None,None,None, None

# def llm_mutate_jointtrain(classifier_bank, img_batch, openai_client, config, iteration):

#     img_batch.to_train()
#     scores = torch.tensor([x[-1] for x in classifier_bank])
#     scores_raw = torch.cat([x[-2][None] for x in classifier_bank],dim=0)
#     classifier_string = [x[0] for x in classifier_bank]
#     classifier_encoded = torch.cat([x[1][None] for x in classifier_bank],dim=0)

#     probabilities = torch.softmax(scores,dim=0)
#     indices = torch.multinomial(probabilities, config.number_of_classifiers_in_prompt, replacement=config.replacement)
    
#     classifier_strings_picked = [classifier_string[i] for i in indices]
#     classifier_encoded_picked = classifier_encoded[indices] 
#     scores_picked = scores[indices].tolist()
#     scores_raw_picked = scores_raw[indices]

#     sorted_indices = sorted(range(len(scores_picked)), key=lambda i: scores_picked[i])

#     scores_picked = [scores_picked[i] for i in sorted_indices]
#     scores_raw_picked = scores_raw_picked[torch.tensor(sorted_indices)] 
#     classifier_strings_picked = [classifier_strings_picked[i] for i in sorted_indices]
#     classifier_encoded_picked = classifier_encoded_picked[torch.tensor(sorted_indices)]

#     try: 
                
#         all_results=[]
#         all_futures=[]
#         list_of_prompts = []
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             num_classes = len(classifier_strings_picked[0])
#             for index in range(num_classes):
#                 print("index: ", index)

#                 theprompt = construct_prompt_jointtrain(classifier_strings_picked, index, iter)
#                 #with open(jointtrain_file + '/prompts.txt', 'a') as t: t.write("iteration: " + str(iter) + "index: " + str(index) + " prompt: " + theprompt + " \n")
#                 list_of_prompts.append(theprompt)
#             futures = [executor.submit(call_OpenSource, list_of_prompts[int(i/config.per_index_gen)], int(i/config.per_index_gen), openai_client) for i in range(config.per_index_gen*num_classes)]
#             all_results = [future.result() for future in concurrent.futures.as_completed(futures)]

#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             #for every time i call vllm, I have a best set. 
#             best_result_futures = [executor.submit(best, img_batch,all_results[i][0],all_results[i][1],classifier_encoded_picked, classifier_strings_picked) for i in range(config.per_index_gen*num_classes)] #len(all_results)
#             best_results =  [future.result() for future in concurrent.futures.as_completed(best_result_futures)]
        
#         for index in range(num_classes):

#             img_batch.to_train()
#             filtered_bestresults = [x for x in best_results if x[0]]
#             index_bestresults = [x for x in filtered_bestresults if x[-1]==index]

#             try:
#                 #Now we are taking the maximum across the batches generated per index
#                 indexscores = [index_bestresults[i][0][0] for i in range(len(index_bestresults))]
#                 bestforindex = indexscores.index(max(indexscores))
#                 [gen_cross_entropy_score, gen_raw_scores], gen_classifier_encoded, gen_classifier_string, gen_beforeandafter, gen_index = index_bestresults[bestforindex]
#             except:
#                 continue

#             classifier_bank.append([gen_classifier_string, gen_classifier_encoded, gen_raw_scores, gen_cross_entropy_score])
#             information = [scores_picked, classifier_encoded_picked, scores_raw_picked, scores_raw, scores, classifier_string, classifier_encoded, gen_classifier_encoded, gen_raw_scores, gen_beforeandafter]
#         return classifier_bank, information 
#     except Exception as error:
#         print(error)
#         return classifier_bank, None
                        
def evaluate_generation_and_log_pretrain(pretraining_file, information, img_batch, dict_of_generated_programs, iteration):
    [theprompt, newbest, scores_all, scores_all_picked, scores_picked, values_picked, scores_pos_generation,scores_mixed_generation,scores, programs] = information
    #We keep track of the maximum score in our prompt.
    best_prompt_score = torch.max(torch.tensor(scores_picked)).item()
    best_prompt_score_index = torch.argmax(torch.tensor(scores_picked))
    best_prompt_program = values_picked[best_prompt_score_index].split("score: ")[0] 

    scores_pos_best_prompt_train, scores_mixed_best_prompt_train = scores_all_picked[best_prompt_score_index][0], scores_all_picked[best_prompt_score_index][1] 

    best_llm_index = torch.argmax(scores).item()
    best_llm_program = programs[best_llm_index]
    best_llm_score=scores[best_llm_index].item()
    scores_pos_best_bank_train, scores_mixed_best_bank_train = scores_all[best_llm_index][0],scores_all[best_llm_index][1]

    #now lets also log the classifier's performance on the test set. 
    img_batch.to_test()

    attributes = extract_attributes_from_string(newbest)
    encoded_attributes = transform_tensor(img_batch.encode(attributes))
    scores_pos_generation_test, scores_mixed_generation_test = img_batch.score(encoded_attributes)

    attributes = extract_attributes_from_string(best_prompt_program)
    encoded_attributes = transform_tensor(img_batch.encode(attributes))
    scores_pos_best_prompt_test, scores_mixed_best_prompt_test = img_batch.score(encoded_attributes)

    attributes = extract_attributes_from_string(best_llm_program)
    encoded_attributes = transform_tensor(img_batch.encode(attributes))
    scores_pos_best_bank_test, scores_mixed_best_bank_test = img_batch.score(encoded_attributes)

    scores_pos_best_prompt_train_old,scores_mixed_best_prompt_train_old = scores_all_picked[best_prompt_score_index][0],scores_all_picked[best_prompt_score_index][1]
    scores_pos_best_bank_train_old, scores_mixed_best_bank_train_old = scores_all[best_llm_index][0],scores_all[best_llm_index][1]


    list_of_train = [scores_pos_generation[None],scores_mixed_generation[None], 
                    scores_pos_best_prompt_train[None],scores_mixed_best_prompt_train[None],
                    scores_pos_best_prompt_train_old[None],scores_mixed_best_prompt_train_old[None],
                    scores_pos_best_bank_train[None], scores_mixed_best_bank_train[None],
                    scores_pos_best_bank_train_old[None], scores_mixed_best_bank_train_old[None]]
    list_of_test = [scores_pos_generation_test[None], scores_mixed_generation_test[None],
                    scores_pos_best_prompt_test[None], scores_mixed_best_prompt_test[None],
                    scores_pos_best_bank_test[None], scores_mixed_best_bank_test[None]] 
    
    scores_train = torch.cat(list_of_train,dim=0)
    scores_test = torch.cat(list_of_test,dim=0)
    
    
    iteration = iteration + 1
    print("iteration: ", iteration)

    ####### LOGGING ########
    if not os.path.isdir(pretraining_file + "/program_bank"): 
        os.mkdir(pretraining_file + "/program_bank")
    with open(pretraining_file + "/program_bank/iter_" + str(iter) +  ".pkl", 'wb') as f: 
        pickle.dump(dict_of_generated_programs, f)

    with open(pretraining_file + '/best_program.txt', 'a') as b:
        b.write("iteration:  " + str(iteration) + " best program: " + best_llm_program.replace("\n","") + " \n")
    with open(pretraining_file + '/prompts.txt', 'a') as t:
        t.write("iteration: " + str(iteration) + " prompt: " + theprompt + " \n" + "generation: " + newbest.replace("\n","") + "\n")
    with open(pretraining_file +'/generations.txt', 'a') as c:
        c.write("iteration: " + str(iteration) + " generation: " + newbest.replace("\n","") + "\n")
    with open(pretraining_file + '/scores.txt', 'a') as s:
        #pdb.set_trace()
        s.write("iteration: " + str(iteration) + " generation: " + str(torch.mean(scores_pos_best_bank_train_old).item()) + "\n")
    return dict_of_generated_programs

                        

def llm_mutate_pretrain(classifier_bank, img_batch, openai_client, config, iteration):
    img_batch.to_train()
    scores_all = [[x[2],x[3]] for x in classifier_bank]
    scores_prompt = [get_meanlist_from_scores(x[2],x[3]) for x in classifier_bank]
    scores = torch.tensor([x[-1] for x in scores_prompt])

    programs = [x[0] for x in classifier_bank]
    probabilities = torch.softmax(scores,dim=0)
    indices = torch.multinomial(probabilities, config.number_of_classifiers_in_prompt, replacement=config.replacement)
    
    values_picked = [programs[i] for i in indices]
    scores_picked = scores[indices].tolist()
    scores_prompt_picked = [scores_prompt[x] for x in indices]
    scores_all_picked = [scores_all[x] for x in indices]

    sorted_indices = sorted(range(len(scores_picked)), key=lambda i: scores_picked[i])

    scores_picked = [scores_picked[i] for i in sorted_indices]
    if not config.random_scores_order:
        values_picked = [values_picked[i] for i in sorted_indices]
    scores_prompt = [scores_prompt_picked[i] for i in sorted_indices]
    scores_all_picked = [scores_all_picked[x] for x in sorted_indices]

    #We construct a prompt out of these "n" programs.
    theprompt = construct_prompt(values_picked, scores_prompt, iteration)
    
    #We ask LLM to run the prompt and create a new program. 
    #pdb.set_trace()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(call_OpenSource, theprompt, 0, openai_client) for _ in range(80)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    #pdb.set_trace()
    try: 
        list_of_scores = []
        list_of_pos_scores = []
        list_of_neg_scores = []
        list_of_completion = []
        list_of_encoding = []
        for i,(elem,_) in enumerate(results):
            completion = extract_newfun_definition(elem, "newfun")
            if completion: 
                try: 
                
                    thelist = extract_attributes_from_string(completion)
                    repeats_check = has_repeats(thelist)
                    if not repeats_check:
                        new_encodings = img_batch.encode(thelist)
                        transformed_encodings = transform_tensor(new_encodings)
                        pos_scores, neg_scores = img_batch.score(transformed_encodings)
                        score = torch.mean(pos_scores-neg_scores).item()
                        list_of_pos_scores.append(pos_scores)
                        list_of_neg_scores.append(neg_scores)
                        list_of_completion.append(completion)
                        list_of_encoding.append(transformed_encodings)
                        list_of_scores.append(score)
                    
                except Exception as error:
                    print(error, "1")
                    #pdb.set_trace()
                    continue
        
        index = list_of_scores.index(max(list_of_scores))
        newbest =  list_of_completion[index]
        newbest_encoding = list_of_encoding[index]
        scores_pos_generation, scores_mixed_generation = list_of_pos_scores[index], list_of_neg_scores[index]
        #mean_list_generation = get_meanlist_from_scores(scores_pos_generation, scores_mixed_generation)
        classifier_bank.append([newbest, newbest_encoding, scores_pos_generation, scores_mixed_generation])
        information = [theprompt, newbest, scores_all, scores_all_picked, scores_picked, values_picked, scores_pos_generation,scores_mixed_generation,scores, programs]
        return classifier_bank, information
    
    except Exception as error:
        print(error, "2")
        #pdb.set_trace()
        return classifier_bank, None 

def llm_mutate_jointtrain(program_bank, num_classes, img_batch_joint, scientific_names, iteration, jointtrain_file, config, client):
    scores = torch.tensor([x[-1] for x in program_bank])
    scores_raw = torch.cat([x[-2][None] for x in program_bank], dim=0)
    programs_string = [x[0] for x in program_bank]
    programs_encoded = torch.cat([x[1][None] for x in program_bank], dim=0)

    probabilities = torch.softmax(scores, dim=0)
    indices = torch.multinomial(probabilities, config.number_of_classifiers_in_prompt, replacement=config.replacement)
    
    programs_strings_picked = [programs_string[i] for i in indices]
    programs_encoded_picked = programs_encoded[indices] 
    scores_picked = scores[indices].tolist()
    scores_raw_picked = scores_raw[indices]

    sorted_indices = sorted(range(len(scores_picked)), key=lambda i: scores_picked[i])
    scores_picked = [scores_picked[i] for i in sorted_indices]
    scores_raw_picked = scores_raw_picked[torch.tensor(sorted_indices)]
    programs_strings_picked = [programs_strings_picked[i] for i in sorted_indices]
    programs_encoded_picked = programs_encoded_picked[torch.tensor(sorted_indices)]

    # Get the best result per class
    best_prompt_score_index = torch.argmax(torch.tensor(scores_picked))
    best_prompt_program = programs_encoded_picked[best_prompt_score_index]

    scores_crossentropy_bestprompt_train = scores_picked[best_prompt_score_index]
    scores_raw_bestprompt_train = scores_raw_picked[best_prompt_score_index]

    best_llm_index = torch.argmax(scores).item()
    best_llm_program_string = programs_string[best_llm_index]
    best_llm_program_encoded = programs_encoded[best_llm_index]

    scores_crossentropy_bestbank_train = scores[best_llm_index].item()
    scores_raw_bestbank_train = scores_raw[best_llm_index]
    accuracy_train_best = get_accuracy(scores_raw_bestbank_train, img_batch_joint.classidx)

    img_batch_joint.to_test()

    scores_crossentropy_bestprompt_test, scores_raw_bestprompt_test = img_batch_joint.score(best_prompt_program)
    scores_crossentropy_bestbank_test, scores_raw_bestbank_test = img_batch_joint.score(best_llm_program_encoded)
    accuracy_test_best = get_accuracy(scores_raw_bestbank_test, img_batch_joint.classidx)

    # Directly call the logging function
    log_and_save_files(iteration, jointtrain_file, scores_raw_bestbank_train, scores_raw_bestprompt_train, scores_raw_bestbank_test, scores_raw_bestprompt_test, 
    scores_crossentropy_bestbank_train, scores_crossentropy_bestbank_test, accuracy_train_best, accuracy_test_best, scientific_names, best_llm_program_string, num_classes)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list_of_prompts = []
        for index in range(num_classes):
            theprompt = construct_prompt_jointtrain(programs_strings_picked, index, iter)
            list_of_prompts.append(theprompt)

        futures = [executor.submit(call_OpenSource, list_of_prompts[int(i / config.per_index_gen)], int(i / config.per_index_gen), client) for i in range(config.per_index_gen * num_classes)]
        all_results = [future.result() for future in concurrent.futures.as_completed(futures)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        #img_batch, result, index, programs_encoded_picked_cur_preclone, programs_strings_picked_cur_preclone
        best_result_futures = [executor.submit(best, img_batch_joint, all_results[i][0], all_results[i][1], programs_encoded_picked, programs_strings_picked) for i in range(config.per_index_gen * num_classes)]
        best_results = [future.result() for future in concurrent.futures.as_completed(best_result_futures)]

    for index in range(num_classes):
        img_batch_joint.to_train()
        filtered_bestresults = [x for x in best_results if x[0]]
        index_bestresults = [x for x in filtered_bestresults if x[-1] == index]

        try:
            indexscores = [index_bestresults[i][0][0] for i in range(len(index_bestresults))]
            bestforindex = indexscores.index(max(indexscores))
            [gen_cross_entropy_score, gen_raw_scores], gen_program_encoded, gen_program_string, gen_beforeandafter, gen_index = index_bestresults[bestforindex]
        except:
            continue

        program_bank.append([gen_program_string, gen_program_encoded, gen_raw_scores, gen_cross_entropy_score])

        if not os.path.isdir(jointtrain_file + "/program_bank"): 
            os.mkdir(jointtrain_file + "/program_bank")
        with open(jointtrain_file + "/program_bank/iter_" + str(iter) +  ".pkl", 'wb') as f:
            pickle.dump(program_bank, f)

        log_generations_and_append_to_bank(jointtrain_file, scientific_names, iteration, index, gen_raw_scores, gen_program_encoded, gen_beforeandafter, 
        gen_cross_entropy_score, img_batch_joint)

    return program_bank

def log_generations_and_append_to_bank(jointtrain_file, scientific_names, iter, index, gen_raw_scores, gen_program_encoded, gen_beforeandafter, gen_cross_entropy_score, img_batch_joint):
    
    accuracy_train_gen = get_accuracy(gen_raw_scores, img_batch_joint.classidx)
    np.save(jointtrain_file + "/scores_train/" + str(iter) +"_" +str(index)+ "_index.npy", img_batch_joint.classidx.detach().numpy())
    np.save(jointtrain_file + "/scores_train/" + str(iter) +"_" +str(index) + "_raw_gen.npy", gen_raw_scores.detach().numpy())

    img_batch_joint.to_test()

    gen_cross_entropy_score_test, gen_raw_scores_test = img_batch_joint.score(gen_program_encoded)
    accuracy_test_gen = get_accuracy(gen_raw_scores_test, img_batch_joint.classidx)
    np.save(jointtrain_file + "/scores_test/" + str(iter) +"_" +str(index)+ "_index.npy", img_batch_joint.classidx.detach().numpy())
    np.save(jointtrain_file + "/scores_test/" + str(iter) +"_" +str(index)+ "_raw_gen.npy", gen_raw_scores_test.detach().numpy())

    with open(jointtrain_file +'/generations.txt', 'a') as c:
        with open(jointtrain_file + '/scores_gen.txt', 'a') as q:
            c.write("iteration: " + str(iter) + " index: " + scientific_names[index] + " generation: " + gen_beforeandafter + "\n")
            q.write("iteration: " + str(iter) + " index: " + scientific_names[index] + " train cross-entropy: " + str(gen_cross_entropy_score) + " test cross-entropy: " + str(gen_cross_entropy_score_test) + " train accuracy: " + str(accuracy_train_gen) + " test accuracy: " + str(accuracy_test_gen) + "\n")
    img_batch_joint.to_train()


def log_and_save_files(iter, jointtrain_file, scores_raw_bestbank_train, scores_raw_bestprompt_train, scores_raw_bestbank_test, scores_raw_bestprompt_test, scores_crossentropy_bestbank_train, scores_crossentropy_bestbank_test, accuracy_train_best, accuracy_test_best, scientific_names, best_llm_program_string, num_classes):
    np.save(jointtrain_file + "/scores_train/" + str(iter) + "_raw_bestbank.npy", scores_raw_bestbank_train.detach().numpy())
    np.save(jointtrain_file + "/scores_train/" + str(iter) + "_raw_bestprompt.npy", scores_raw_bestprompt_train.detach().numpy())
    np.save(jointtrain_file + "/scores_test/" + str(iter) + "_raw_bestbank.npy", scores_raw_bestbank_test.detach().numpy())
    np.save(jointtrain_file + "/scores_test/" + str(iter) + "_raw_bestprompt.npy", scores_raw_bestprompt_test.detach().numpy())

    with open(jointtrain_file + '/best_program.txt', 'a') as b:
        with open(jointtrain_file + '/scores_best.txt', 'a') as s:
            s.write("iteration: " + str(iter) + " train cross-entropy: " + str(scores_crossentropy_bestbank_train) + " test cross-entropy: " + str(scores_crossentropy_bestbank_test) + " train accuracy: " + str(accuracy_train_best) + " test accuracy: " + str(accuracy_test_best) + "\n")
            perclass = "".join(["class " + scientific_names[i] + ": " + str(best_llm_program_string[i]) + "\n" for i in range(num_classes)])
            b.write("iteration: " + str(iter) + " best program: \n" + perclass)

def populate_classifier_bank(imgbatch, num_init_bank, allattributes, pretrain=False, num_classes=5):
   
    classifier_bank = []

    for i in tqdm(list(range(num_init_bank))): 

        if pretrain:
            string_attributes = np.random.choice(allattributes,6).tolist() 
            program = "def newfun(image): return image.score(" 
            program= program + str(string_attributes) + ") \n"
            attribute_encodings = transform_tensor(imgbatch.encode(string_attributes))
            scores_pos, scores_mixed = imgbatch.score(attribute_encodings)
            classifier_bank.append([program,attribute_encodings,scores_pos, scores_mixed])

        else:
            list_of_class_attributes=[]
            list_of_class_attributes_encoded = []

            for j in range(num_classes):
                string_attributes = np.random.choice(allattributes[j],6,replace=False).tolist() 
                attribute_encodings = transform_tensor(imgbatch.encode(string_attributes))
                list_of_class_attributes_encoded.append(attribute_encodings.T[None])
                list_of_class_attributes.append(string_attributes)
            
            encoded_class_attributes=torch.cat(list_of_class_attributes_encoded,dim=0)
            cross_entropy_score, raw_scores = imgbatch.score(encoded_class_attributes)
            classifier_bank.append([list_of_class_attributes,encoded_class_attributes, raw_scores,cross_entropy_score])


    if pretrain:
        sorted_scores = sorted(classifier_bank, key=lambda x: get_meanlist_from_scores(x[2],x[3])[-1], reverse=True)
    else:
        sorted_scores = sorted(classifier_bank, key=lambda x: [-1], reverse=True)

    classifier_bank = list(sorted_scores[:30])

    return classifier_bank
    
def load_data(path_to_imagenet_id, path_to_imagenet_decriptors, path_to_families):
    with open(path_to_imagenet_id, 'r') as f:
        data = f.read()
        dictionary_data = ast.literal_eval(data)
        id_dict = {i: 'n'+dictionary_data[i]['id'][:8] for i in dictionary_data}

    with open(path_to_imagenet_decriptors, 'r') as file:
        gpt_descriptions = json.load(file)

    with open(path_to_families, 'r') as f:
        families = json.load(f)

    totallist = [value for key,value in gpt_descriptions.items()]
    allattributes = [escape_quotes(x) for xs in totallist for x in xs]

    return id_dict, allattributes, families


def load_imagenet_dicts():
    #dictionary of index : class synthset id

    with open('imagenet_label_to_wordnet_synset.txt', 'r') as f:
        data = f.read()
        dictionary_data = ast.literal_eval(data)
        id_dict = {i: 'n'+dictionary_data[i]['id'][:8] for i in dictionary_data}
        opposite_id_dict = {'n'+dictionary_data[i]['id'][:8]: i for i in dictionary_data}

    #0-shot descriptors
    with open('descriptors_imagenet.json', 'r') as file:
        gpt_descriptions = json.load(file)

    #dictionary of index : class string
    dict_of_classtring = {i: key for i,key in enumerate(gpt_descriptions.keys())}
    return id_dict, opposite_id_dict, gpt_descriptions, dict_of_classtring


def construct_prompt(descriptors, scores_picked, iter): #Here are some programs in increasing order of how well an attribute matches a set of images. We are playing a game of attribute discovery. 
    if iter%2==0:
        #theprompt = "Here are some programs in increasing order of how good the attributes are at classifying an image for a class. The parenthesis next to each attribute is the score for that attribute. The programs are ranked according to average accuracy. We are playing a game of attribute discovery. Based on what youve seen below, propose a new program, called 'newfun', with exactly 6 visual attributes in it, that you think might achieve an even higher score. Try to use new visual attributes when generating the new program, not just synonyms of previous attributes: " #Based on what you've seen below, propose a new program, with a new unseen attribute in it, that you think would achieve an even higher accuracy from what you've seen below. Propose programs with attributes that are not repeats or synonyms of existing attributes: "
        theprompt = "Here are some programs in increasing order of how good the attributes are at classifying an image for a class. The parenthesis next to each attribute is the score for that attribute. The programs are ranked according to average accuracy. We are playing a game of attribute discovery. Based on what youve seen below, propose a new program, called 'newfun', with exactly 6 visual attributes in it, that you think might achieve an even higher score. Try to use nouns when generating the new program: " #Based on what you've seen below, propose a new program, with a new unseen attribute in it, that you think would achieve an even higher accuracy from what you've seen below. Propose programs with attributes that are not repeats or synonyms of existing attributes: "
    else:
        theprompt = "Here are some programs in increasing order of how good the attributes are at classifying an image for a class. The parenthesis next to each attribute is the score for that attribute. The programs are ranked according to average accuracy. We are playing a game of attribute discovery. Based on what youve seen below, propose a new program, called 'newfun', with exactly 6 visual attributes in it, that you think might achieve an even higher score. "

    for i,description in enumerate(descriptors):
        #pdb.set_trace()
        test= description.split('\'')
        indices = [x for x in range(1,len(test),2)]
        for j,ind in enumerate(indices):
            try:
                test[ind]=test[ind] #+ " (" + str(round(scores_picked[i][j],5)) + ")"
            except:
                pdb.set_trace()

        theprompt = theprompt + '\''.join(test) + " (mean score: " + str(round(scores_picked[i][-1],5)) + ")" # + ")" # + " score per attribute: "  + str(scores_picked[i][:6]) + " mean score: " + str(scores_picked[i][-1]) + " \n " 

    return theprompt

def construct_prompt_jointtrain(descriptors, classidx, iter): #Here are some programs in increasing order of how well an attribute matches a set of images. We are playing a game of attribute discovery. 

    theprompt = "Here are some programs for class X. The programs are ranked according to average accuracy. We are playing a game of attribute discovery. Based on what youve seen below, propose a new program with diverse visual attributes that you think might achieve an even higher score. Please try to make new original attributes out of what you have seen, instead of just repeating. \n"

    for i,description in enumerate(descriptors):
        current = description[classidx] 

        
        program = " def newfun(image): return image.score(" 
        program= program + str(current) + ") \n"
        if program not in theprompt:
            theprompt = theprompt + " class " + str(classidx) + program
 
    return theprompt