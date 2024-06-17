import pdb
from config import Config
from environment_setup import initialize_model, initialize_llm
from utilities import *
from image_batch import ImageBatch
from data_loader import load_datasets_pretrain, load_datasets_jointtrain

animal_classes = ['00403_Animalia_Arthropoda_Insecta_Coleoptera_Scarabaeidae_Trypoxylus_dichotomus',
                                        '03127_Animalia_Chordata_Aves_Accipitriformes_Accipitridae_Buteo_jamaicensis',
                                        '05328_Animalia_Mollusca_Gastropoda_Nudibranchia_Polyceridae_Triopha_maculata',
                                        '01978_Animalia_Arthropoda_Insecta_Lepidoptera_Nymphalidae_Tegosa_claudina',
                                        '06344_Plantae_Tracheophyta_Liliopsida_Poales_Poaceae_Bromus_catharticus']

def main():
    config = Config()
    model, preprocess = initialize_model(config.scoring)
    openai_client = initialize_llm(config.llm_type)

    
    
    # Perform pre-training, or load in pre-training from previous run 
    if config.pretraining_path==None:
        list_of_pretraining_files = pretraining(openai_client,model,config)
    else:
        list_of_pretraining_files = [x + config.pretraining_path + "/experiment0/" for x in os.listdir(config.pretraining_path)]
    
    jointtrain_file = joint_training(openai_client, model, config, list_of_pretraining_files)
    print("Final results in: ", jointtrain_file)
    

def joint_training(openai_client, model, config, list_of_pretraining_files):

    expname = 'jointtrain_after_binaryclassifier_' + str(config.number_of_classifiers_in_prompt) + "_exinprompt"
    jointtrain_file=create_folder_and_log_exp_details(expname, config.llm_type, config.scoring,config.append_class,config.objective,config.synset, config.batch_size)

    # Initialize ImageBatch for handling image operations
    img_batch = ImageBatch(model, list_of_pretraining_files, config.hparams['image_size'], config.scoring, pretrain=False)

    # Load init attributes from pretraining
    joint_train_init_attributes = []
    
    for classname in list_of_pretraining_files:
        pretraining_file = classname 
        with open(pretraining_file + "/generations.txt", 'r') as f:
            text = f.read()
        words = re.findall(r"'([^']*)'", text)
        new_attributes = [x for x in words if "newfun" not in x and len(x) > 5] 
        joint_train_init_attributes.append(new_attributes)

    inat_dataloader = load_datasets_jointtrain(config.dataset_path, config.synset_ids, config.batch_size)
    dict_of_generated_programs={}
    
    #Let's pre-encode the entire dataset for both the training set and the test set as a single batch (this fits into memory since we never use gradients)
    for pos_images, class_idx in inat_dataloader['train']:
        img_batch.reinit_images_jointtrain(pos_images, class_idx, train=True)
        img_batch.to_train()

    for images,class_idx in inat_dataloader['test']:
        img_batch.reinit_images_jointtrain(images,class_idx,train=False)

    classifier_bank = populate_classifier_bank(img_batch, config.classifiers_initialized, joint_train_init_attributes, pretrain=False, num_classes=len(list_of_pretraining_files))
    
    num_classes = len(list_of_pretraining_files)
    for iteration in range(config.max_iter):
        dict_of_generated_programs, classifier_bank = process_iteration(iteration, list_of_pretraining_files, num_classes, jointtrain_file, config, classifier_bank, img_batch, openai_client, dict_of_generated_programs, pretrain=False)
        if iteration == config.max_iter - 1:
            print("Reached the maximum number of iterations.")
            break

    return jointtrain_file

def pretraining(openai_client,model,config):

    

    # Load description data and iNat dataset paths
    id_dict, init_attributes, families = load_data('files/imagenet_label_to_wordnet_synset.txt', 'files/descriptors_imagenet.json', 'files/inaturalist_species.json')
    scientific_names = families[config.synset]["classes"]

    # Initialize ImageBatch for handling image operations
    img_batch = ImageBatch(model, scientific_names, config.hparams['image_size'], config.scoring, pretrain=True)

    expname = 'binaryclassifier_' + config.dataset_name + "_" + config.synset + '_' + str(config.number_of_classifiers_in_prompt) + 'prompt'
    list_of_pretraining_files=[]
        
    for i in range(0,len(scientific_names)):

        scientific_names = families[config.synset]["classes"]
        main_class = [scientific_names[i]]
        scientific_names_duplicate = scientific_names.copy()
        scientific_names_duplicate.remove(scientific_names[i])
        rest_classes = animal_classes + scientific_names_duplicate
        inat_dataloader = load_datasets_pretrain(config.dataset_path, main_class, rest_classes, config.batch_size, transform=None)
        
        dict_of_generated_programs={}

        #Let's pre-encode the entire dataset for both the training set and the test set as a single batch (this fits into memory since we never use gradients)
        for pos_images, neg_images in inat_dataloader['train']:
            img_batch.reinit_images_pretrain(pos_images, neg_images, train=True)
            img_batch.to_train()

        for images,class_idx in inat_dataloader['test']:
            img_batch.reinit_images_pretrain(images,class_idx,train=False)

        classifier_bank = populate_classifier_bank(img_batch, config.classifiers_initialized, init_attributes, pretrain=True)
        pretraining_file = create_folder_and_log_exp_details(expname, config.llm_type, config.scoring,config.append_class,config.objective,scientific_names[i], config.batch_size)

        for iteration in range(config.max_iter):
            pdb.set_trace()
            dict_of_generated_programs, classifier_bank = process_iteration(iteration, None, len(scientific_names), pretraining_file, config, classifier_bank, img_batch, openai_client, dict_of_generated_programs, pretrain=True)

            if iteration == config.max_iter - 1:
                print("Reached the maximum number of iterations.")
                break

        list_of_pretraining_files.append(pretraining_file)

    return list_of_pretraining_files

#process_iteration(iteration, list_of_pretraining_files, num_classes, jointtrain_file, config, classifier_bank, img_batch, openai_client, dict_of_generated_programs, pretrain=False)
def process_iteration(iteration, scientific_names, num_classes, file, config, classifier_bank, img_batch, openai_client, dict_of_generated_programs, pretrain):
    
    # Simulate classifier enhancement and evaluation process
    try:
        print(iteration, "here")
        if pretrain:
            classifier_bank, information = llm_mutate_pretrain(classifier_bank, img_batch, openai_client, config, iteration)
        else:
            classifier_bank = llm_mutate_jointtrain(classifier_bank, num_classes, img_batch, scientific_names, iteration, file, config, openai_client) #, information = llm_mutate_jointtrain(classifier_bank, img_batch, openai_client, config, iteration)
        #pdb.set_trace()
        if pretrain and information!=None:
            #if pretrain:
            dict_of_generated_programs = evaluate_generation_and_log_pretrain(file, information, img_batch, dict_of_generated_programs, iteration)
                #llm_mutate_jointtrain(program_bank, num_classes, img_batch_joint, train_dataset_joint, iteration, jointtrain_file, config, client)
                #dict_of_generated_programs = evaluate_generation_and_log_jointtrain(pretraining_file, num_classes, information, classifier_bank, img_batch, dict_of_generated_programs, iteration)
        return dict_of_generated_programs, classifier_bank
    
    except Exception as e:
        print(f"Error processing classifier bank at iteration {iteration} and the error is: {e}")
        return dict_of_generated_programs, classifier_bank

if __name__ == "__main__":
    main()