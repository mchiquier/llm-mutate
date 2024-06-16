# Evolving Interpretable Visual Classifiers with Large Language Models

This is the code for the paper [Evolving Interpretable Visual Classifiers with Large Language Models](https://llm-mutate.cs.columbia.edu/) by [Mia Chiquier](https://www.cs.columbia.edu/~mia.chiquier/)\, [Utkarsh Mall](https://www.cs.columbia.edu/~utkarshm/)\ and [Carl Vondrick](https://www.cs.columbia.edu/~vondrick/). 


![teaser](teaser.gif "Teaser")

## Quickstart
Clone recursively:
```bash
git clone --recurse-submodules https://github.com/mchiquier/llm-mutate.git
```

After cloning:
```bash
cd llm-mutate
export PATH=/usr/local/cuda/bin:$PATH
bash setup.sh  # This may take a while. Make sure the vipergpt environment is active
cd GLIP
python setup.py clean --all build develop --user
cd ..
echo YOUR_OPENAI_API_KEY_HERE > api.key
```

## Detailed Installation
1. Clone this repository with its submodules.
2. Install the dependencies. See the see [Dependencies](#Dependencies).
3. Download two pretrained models (the rest are downloaded automatically). See [Pretrained models](#Pretrained-models).
4. Set up the OpenAI key. See [OpenAI key](#OpenAI-key).

### Cloning this Repo

```bash
git clone --recurse-submodules [https://github.com/cvlab-columbia/viper.git](https://github.com/mchiquier/llm-mutate.git)
```

### Dependencies

First, create a conda environment using `setup_env.sh`. 
To do so, just `cd` into the `llm-mutate` directory, and run:

```bash
export PATH=/usr/local/cuda/bin:$PATH
bash setup_env.sh
conda activate llmmutate
```

### Pretrained models


#### vLLM

openai key

## Datasets

## Configuration


## Running LLM-Mutate on a dataset

## Inference


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
