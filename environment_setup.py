import torch
from clip import load as clip_load
from archive.xvlm import XVLMModel
from openai import OpenAI

def initialize_llm(llm_type):
    openai_api_key = "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"
    if llm_type == 'llama':
        return OpenAI(api_key=openai_api_key, base_url="http://localhost:8000/v1")
    else:
        return OpenAI(api_key=openai_api_key)
    
def initialize_model(scoring):
    """
    Initialize the model based on the specified scoring type.
    
    Args:
    scoring (str): The scoring method to use ('clip' or another type).

    Returns:
    tuple: A tuple containing the initialized model and the preprocessing function.
    """
    # Define hyperparameters dictionary
    hparams = {}

    # Check the scoring type and initialize the model accordingly
    if scoring == 'clip':
        # For 'clip' scoring, use specific model size and device
        hparams['image_size'] = 224
        hparams['device'] = 'cpu'  # Replace 'cpu' with 'cuda' if GPU is available and desired
        hparams['model_size'] = "ViT-B/32"

        # Load the CLIP model from OpenAI CLIP implementation
        device = torch.device(hparams['device'])
        model, preprocess = clip_load(hparams['model_size'], device=device, jit=True)
    
    else:
        # For other scoring, an example using XVLM
        hparams['image_size'] = 384
        hparams['device'] = 'cpu'  # Replace 'cpu' with 'cuda' if GPU is available and desired

        # Initialize the XVLM model (assuming similar interface)
        model = XVLMModel()
        preprocess = None  # Assume no preprocessing is needed or done differently

    return model, preprocess