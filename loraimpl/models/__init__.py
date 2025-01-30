from loraimpl.models.lora_gpt2 import GPT2LMHeadModelLora
from loraimpl.models.bareFinetuning_gpt2 import GPT2LMHeadModelFinetuning
from transformers import GPT2LMHeadModel

gpt2_modifications = {
    'lora': GPT2LMHeadModelLora,
    'adapter': GPT2LMHeadModelFinetuning,
    'none': GPT2LMHeadModel
}