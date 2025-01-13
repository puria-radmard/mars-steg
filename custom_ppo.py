import os
import shutil
import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import (
    ModelConfig,
    PPOConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from custom_ppo_trainer import CustomPPOTrainer as PPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from torch.utils.data import DataLoader
# Definir dispositivo y número de workers
num_cores = 1
if torch.cuda.is_available():
    name_device = 'cuda'
    num_workers = num_cores
elif torch.backends.mps.is_available():
    name_device = 'mps'
    num_workers = num_cores
else:
    name_device = 'cpu'
    num_workers = 0

# Definir campo de texto del dataset
dataset_text_field = "prompt"  # Ajustar según el dataset utilizado

# Función para preprocesar el dataset
def prepare_dataset(dataset, tokenizer, max_length=128):
    """
    Pre-tokeniza el dataset antes de entrenar.
    """
    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],  # Campo del texto
            truncation=True,
            max_length=max_length,
            padding="max_length",
             return_tensors="pt"
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,  # Eliminar columnas originales
        num_proc=1,  # Ajustar si es necesario
    )

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Borrar carpeta de salida si existe
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    quantization = 0  # Bandera para usar cuantización

    ################
    # Modelo y Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    if quantization:
        quantization_config = get_quantization_config(model_args)
        model_kwargs = dict(
            revision=model_args.model_revision,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
    else:
        model_kwargs = dict(
            revision=model_args.model_revision,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token 
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # Cargar modelos
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    peft_config = get_peft_config(model_args)
    ref_policy = None
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    )
    eval_samples = 100

    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

    
    def preprocess_data(batch):
        return tokenizer(batch[dataset_text_field], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(preprocess_data, batched=True)
    data_loader = DataLoader(tokenized_dataset, batch_size=1, shuffle=True)
        

  
    # Definir recompensa
    def compute_reward(inputs, outputs):
        return torch.tensor([1.0] * len(inputs)).to(outputs.device)  # Recompensa simple como placeholder

    ################
    # Entrenamiento
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        for batch in data_loader:
            print(batch)

            input_ids = torch.as_tensor(batch['input_ids']).to(name_device)
            attention_mask = torch.as_tensor(batch['attention_mask']).to(name_device)
            print(input_ids.shape)
            print(attention_mask.shape)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            outputs = policy.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id,
            )

            rewards = compute_reward(input_ids, outputs)
            stats = trainer.step(input_ids, outputs, rewards)
            print(stats)

        # Guardar modelo al final de cada época
        policy.save_pretrained(f"ppo_gpt2_epoch_{epoch + 1}")
        tokenizer.save_pretrained(f"ppo_gpt2_epoch_{epoch + 1}")
