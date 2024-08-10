# Evolving Interpretable Visual Classifiers with Large Language Models (ECCV 2024)

This is the code for the paper [Evolving Interpretable Visual Classifiers with Large Language Models](https://llm-mutate.cs.columbia.edu/) by [Mia Chiquier](https://www.cs.columbia.edu/~mia.chiquier/), [Utkarsh Mall](https://www.cs.columbia.edu/~utkarshm/) and [Carl Vondrick](https://www.cs.columbia.edu/~vondrick/). 


![teaser](teaser.jpg "Teaser")

## Quickstart
Clone recursively:
```bash
git clone --recurse-submodules https://github.com/mchiquier/llm-mutate.git
```

After cloning:
```bash
cd llm-mutate
export PATH=/usr/local/cuda/bin:$PATH
bash setup.sh  # This may take a while. Make sure the llm-mutate environment is active
cd GLIP
python setup.py clean --all build develop --user
cd ..
echo YOUR_OPENAI_API_KEY_HERE > api.key
```

## Detailed Installation
1. Clone this repository with its submodules.
2. Install the dependencies. See the see [Dependencies](#Dependencies).
3. Install vLLM. See [vLLM](#vLLM).

### Cloning this Repo

```bash
git clone --recurse-submodules https://github.com/mchiquier/llm-mutate.git
```

### Dependencies

First, create a conda environment using `setup_env.sh`. 
To do so, just `cd` into the `llm-mutate` directory, and run:

```bash
export PATH=/usr/local/cuda/bin:$PATH
bash setup_env.sh
conda activate llmmutate
```

### vLLM

Launch a server that hosts the LLM with vLLM in a tmux.
``` 
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-70b-chat-hf --tensor-parallel-size 8 --chat-template ./examples/template_chatml.jinja --trust-remote-code
```

## Datasets

Download the iNaturalist dataset (specifically, 'train' and 'validation') from : https://github.com/visipedia/inat_comp/tree/master/2021 
Update the path to the parent folder of train and val in `config.py` : self.dataset_path = YOUR_PATH.

Download the two KikiBouba datasets here:

KikiBouba_v1: https://drive.google.com/file/d/1-a4FRS9N1DLf3_YIYq8150zN1uilapft/view?usp=sharing
KikiBouba_v2: https://drive.google.com/file/d/17ibF3tzFiZrMb9ZnpYlLEh-xmWkPJpNH/view?usp=sharing

Remember to update the `dataset_path` in the `config.py` file.

## Running LLM-Mutate on iNaturalist

Running this will automatically launch both pre-training and joint-training on the iNaturalist dataset for the `Lichen` synset. To change the synset, modify the config file. See paper for explanation of pretraining/joint training. 
```
python src/llm-mutate.py
```

## Running LLM-Mutate on YOUR dataset!

All you need to do to run LLM-Mutate on your dataset is update the dataset path in the config.py file and also make sure that this path points to a folder that has the following folder structure. 

```
dataset-root/
│
├── train/
│   ├── class 1/
│   │   ├── img1.png 
│   │   ├── img2.png 
...
├── val/
│   ├── class 1/
│   │   ├── img1.png 
│   │   ├── img2.png 
```

NOTE: Make sure to set `do_pretraining=True` if you'd like to do employ pre-training (useful if your dataset is fine-grained classification), otherwise, the default of `do_pretraining=False` is fine.

```
python src/llm-mutate.py
```

## Inference

Specify what experiment you'd like to evaluate in the `config.py` file in the `experiment` attribute. You can pick from: `zero_shot`, `clip_scientific`, `clip_common`, `cbd_scientific`, `cbd_common`, `ours`. Note that for the KikiBouba datasets there is no difference between `scientific` and `common` as there is only one class name per class. 
```
python src/inference.py
```

## Citation

If you use this code, please consider citing the paper as:

```
@article{chiquier2024evolving,
  title={Evolving Interpretable Visual Classifiers with Large Language Models},
  author={Chiquier, Mia and Mall, Utkarsh and Vondrick, Carl},
  journal={arXiv preprint arXiv:2404.09941},
  year={2024}
}
```
