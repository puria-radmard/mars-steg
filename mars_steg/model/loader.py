import torch
from mars_steg.config import ModelConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftMixedModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from typing import Dict

DEFAULT_DTYPE = torch.bfloat16

def load_quantized_model_on_device(model_name, device_id, quantization_config=None):
    """
    Load a quantized model on a specific GPU device while handling quantization configs correctly.
    
    Args:
        model_name (str): HuggingFace model identifier
        device_id (int): GPU device ID
        quantization_config (BitsAndBytesConfig, optional): Quantization configuration
        
    Returns:
        AutoModelForCausalLM: Loaded and quantized model on specified device
    """
    # Set the target device before model initialization
    torch.cuda.set_device(device_id)
    
    # If using quantization, we need to ensure compute dtype matches the device
    if quantization_config:
        # Ensure compute dtype is set appropriately for the device
        if not hasattr(quantization_config, 'bnb_4bit_compute_dtype'):
            quantization_config.bnb_4bit_compute_dtype = DEFAULT_DTYPE
            
        # Set torch_dtype to match compute_dtype
        torch_dtype = quantization_config.bnb_4bit_compute_dtype
    else:
        torch_dtype = DEFAULT_DTYPE  # default dtype
    
    # Load the model with device specification
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=f"cuda:{device_id}",
        torch_dtype=torch_dtype
    )
    
    return model



def get_model(model_config: ModelConfig, tokenizer: AutoTokenizer) -> PeftModel | PeftMixedModel | AutoModelForCausalLM:
    """
    Retrieves and initializes a model based on the provided configuration.

    This function loads a pre-trained causal language model (from GPT-2 by default), resizes its token 
    embeddings to match the tokenizer's vocabulary, and, if specified, applies Low-Rank Adaptation (LoRA) 
    for efficient fine-tuning.

    Parameters
    ----------
    model_config : ModelConfig
        The configuration object containing model-related parameters such as model name and LoRA settings.
    tokenizer : AutoTokenizer
        The tokenizer used to process text inputs.

    Returns
    -------
    model : transformers.AutoModelForCausalLM
        The initialized and configured language model, potentially with LoRA applied.
    """
    # Load the model from pre-trained weights
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name)
    
    # Resize token embeddings to match tokenizer's vocabulary
    model.resize_token_embeddings(len(tokenizer))
    
    # Apply LoRA if enabled in the model configuration
    if model_config.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, **model_config.lora_params
        )
        
        # Apply LoRA to the model and print trainable parameters
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


if __name__ == "__main__":

    # Example configuration for the model
    model_config = ModelConfig(
        model_name="gpt2",  # The base model to use (GPT-2 in this case)
        lora=True,           # Enable LoRA
        lora_r=16,           # Rank for the LoRA layers
        lora_alpha=32,       # Scaling factor for LoRA layers
        lora_dropout=0.1,    # Dropout probability for LoRA layers
        lora_bias="none",    # Bias setting for LoRA
    )

    # Load the tokenizer corresponding to the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    
    # Retrieve and configure the model
    model = get_model(model_config, tokenizer)
    
    # Print out the model's architecture
    print(model)