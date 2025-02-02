from mars_steg.config import ModelConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


def get_model(model_config: ModelConfig, tokenizer: AutoTokenizer):
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