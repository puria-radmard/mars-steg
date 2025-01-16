from mars_steg.config import ModelConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


def get_model(model_config: ModelConfig, tokenizer: AutoTokenizer):
    """Get a model."""
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name)
    model.resize_token_embeddings(len(tokenizer))
    if model_config.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, **model_config.lora_params
        )
        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()
    return model


if __name__ == "__main__":

    model_config = ModelConfig(
        model_name="gpt2",
        lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_bias="none",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    model = get_model(model_config, tokenizer)
    print(model)
