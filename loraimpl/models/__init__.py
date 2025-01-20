from lora_gpt2 import GPT2LMHeadModelLora
from bareFinetuning_gpt2 import GPT2LMHeadModelFinetuning
from transformers import GPT2LMHeadModel

gpt2_modifications = {
    'lora': GPT2LMHeadModelLora,
    'adapter': GPT2LMHeadModelLora,
    'none': GPT2LMHeadModel
}