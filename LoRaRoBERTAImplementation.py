# Basic Imports
import math
import torch
from pathlib import Path
from typing import Union, Dict

import pathlib
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn

# Basic Model Imports
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaModel

# Import Attentions to adjust
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention



class LoRALinearRoberta(RobertaSelfAttention):
    def __init__(self, rank=8, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dimension = self.all_head_size

        self.rank = rank
        self.alpha = alpha

        # Keep the original query, key, and value layers
        # They're already initialized by the parent class

        # Initialize LoRA matrices for query
        std_dev = 1 / torch.sqrt(torch.tensor(rank, dtype=torch.float32))
        self.lora_query_A = nn.Parameter(torch.randn(dimension, rank) * std_dev)
        self.lora_query_B = nn.Parameter(torch.zeros(rank, dimension))

        # Initialize LoRA matrices for value
        self.lora_value_A = nn.Parameter(torch.randn(dimension, rank) * std_dev)
        self.lora_value_B = nn.Parameter(torch.zeros(rank, dimension))

        # Freeze the original parameters
        self.query.weight.requires_grad_(False)
        self.query.bias.requires_grad_(False)
        self.value.weight.requires_grad_(False)
        self.value.bias.requires_grad_(False)

    def lora_query(self, x):
        # Original query transformation
        base_output = self.query(x)
        # LoRA update
        lora_output = self.alpha * (x @ self.lora_query_A @ self.lora_query_B)
        return base_output + lora_output

    def lora_value(self, x):
        # Original value transformation
        base_output = self.value(x)
        # LoRA update
        lora_output = self.alpha * (x @ self.lora_value_A @ self.lora_value_B)
        return base_output + lora_output

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        """Modified forward pass using LoRA for query and value computations"""

        mixed_query_layer = self.lora_query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class FinetuneRoberta(RobertaSelfAttention):
    def __init__(self, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dimension = self.all_head_size

        self.alpha = alpha

        # Keep the original query, key, and value layers
        # They're already initialized by the parent class

        # Initialize LoRA matrices for query
        std_dev = 1 / torch.sqrt(torch.tensor(rank, dtype=torch.float32))
        self.finetune_query = nn.Parameter(torch.randn(dimension, dimension) * std_dev)

        # Initialize LoRA matrices for value
        self.finetune_value = nn.Parameter(torch.randn(dimension, dimension) * std_dev)

        # Freeze the original parameters
        self.query.weight.requires_grad_(False)
        self.query.bias.requires_grad_(False)
        self.value.weight.requires_grad_(False)
        self.value.bias.requires_grad_(False)

    def finetune_query(self, x):
        # Original query transformation
        base_output = self.query(x)
        # LoRA update
        finetune_output = self.alpha * (x @ self.finetune_query)
        return base_output + finetune_output

    def finetune_value(self, x):
        # Original value transformation
        base_output = self.value(x)
        # LoRA update
        finetune_output = self.alpha * (x @ self.finetune_value)
        return base_output + finetune_output

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        """Modified forward pass using LoRA for query and value computations"""

        mixed_query_layer = self.finetune_query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.finetune_value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.finetune_value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.finetune_value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

# The testing!!!!

def count_parameters(model):
    """
    Counts the number of trainable parameters of a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model for which the number of trainable parameters will be counted.

    Returns
    -------
    int
        The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_with_underscore(n):
    """Mini helper function to format a number with underscore as thousand separator"""
    return f"{n:_}"


class LoraWrapperRoberta(nn.Module):
    def __init__(self, task_type, num_classes: int = None, dropout_rate=0.1, model_id: str = "roberta-base",
                 lora_rank: int = 8, train_biases: bool = True, train_embedding: bool = False,
                 train_layer_norms: bool = True):
        """
        Initializes a LoraWrapperRoberta instance, which is a wrapper around the RoBERTa model incorporating
        Low-Rank Adaptation (LoRA) to efficiently retrain the model for different NLP tasks such as GLUE benchmarks
        and SQuAD. LoRA allows for effective adaptation of large pre-trained models like RoBERTa with minimal updates.

        Parameters
        ----------
        task_type : str
            Type of task to configure the model for. Should be one of {'glue', 'squad_v1', 'squad_v2'}.
            For 'squad_v1' and 'squad', the number of classes is set to 2, and for 'squad_v2', it's set to 3,
            to accommodate the "no answer possible" scenario.
        num_classes : int, optional
            The number of classes for the classification layer on top of the RoBERTa model. The default value is
            determined by the task type if not provided. Has to be provided for the glue task.
        dropout_rate : float, default 0.1
            Dropout rate to be used in the dropout layers of the model.
        model_id : str, default "roberta-base"
            Identifier for the pre-trained RoBERTa model to be loaded.
        lora_rank : int, default 8
            Rank of the adaptation applied to the attention layers via LoRA.
        train_biases : bool, default True
            Flag indicating whether to update bias parameters during training.
        train_embedding : bool, default False
            Flag indicating whether to update embedding layer weights during training.
        train_layer_norms : bool, default True
            Flag indicating whether to update the layer norms during training. Usually this is a good idea.

        Examples
        --------
        To initialize a model for the 'glue' task type:

            model = LoraWrapperRoberta(task_type='squad_v1')
        """
        super().__init__()

        supported_task_types = ['glue', 'squad', 'squad_v1', 'squad_v2']
        assert isinstance(task_type,
                          str) and task_type.lower() in supported_task_types, f"task_type has to be one of {supported_task_types}"

        # for squad v1 the num_classes should be 2, for squad v2 it should be 3 (the third is there to predict "no answer possible")
        if task_type == "squad_v1" or task_type == "squad":
            num_classes = 2
        elif task_type == "squad_v2":
            num_classes = 3

        # 1. Initialize the base model with parameters
        self.model_id = model_id
        self.tokenizer = RobertaTokenizer.from_pretrained(model_id)
        self.model = RobertaModel.from_pretrained(model_id)

        self.model_config = self.model.config  # save model config to use when setting the layers

        self.base_model_param_count = count_parameters(self.model)

        self.lora_rank = lora_rank
        self.train_biases = train_biases
        self.train_embeddings = train_embedding
        self.train_layer_norms = train_layer_norms

        # 2. Add the layer for the benchmark tasks
        # Get the output size of the base model & save other parameters
        d_model = self.model_config.hidden_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.task_type = task_type.lower()

        # Define the additional norm, linear layer, and dropout
        self.finetune_head_norm = nn.LayerNorm(d_model)
        self.finetune_head_dropout = nn.Dropout(dropout_rate)
        self.finetune_head_classifier = nn.Linear(d_model, num_classes)

        # 3. set up the lora model for training in Benchmark task:
        self.replace_multihead_attention()
        self.freeze_parameters_except_lora_and_bias()

    def replace_multihead_attention(self, verbose=True):
        """
        Replaces RobertaSelfAttention with LoraRobertaSelfAttention in the model, which contains the LoRA logic and parameters.

        Parameters
        ----------
        verbose : bool
            Whether to print the number of replaced modules. Defaults to True.
        """
        self.nr_replaced_modules = 0
        self.replace_multihead_attention_recursion(self.model)
        if verbose:
            print(f"Replaced {self.nr_replaced_modules} modules of RobertaSelfAttention with LoraRobertaSelfAttention")

    def replace_multihead_attention_recursion(self, model):
        """
        Recursively replaces RobertaSelfAttention with LoraRobertaSelfAttention in the given model/module.
        If some components are wrapped in another class this function can recursively apply the replacement to
        find all instances of the Attention.

        Parameters
        ----------
        model : nn.Module
            The PyTorch module (or full model) to modify.
        """
        # Model can also be a module if it contains sub-components
        for name, module in model.named_children():

            if isinstance(module, RobertaSelfAttention):
                self.nr_replaced_modules += 1

                # Create a new LoraMultiheadAttention layer
                #new_layer = LoRALinearRoberta(rank=self.lora_rank, config=self.model_config)
                new_layer = FinetuneRoberta(config=self.model_config)

                # Get the state of the original layer
                state_dict_old = module.state_dict()

                # Load the state dict to the new layer
                new_layer.load_state_dict(state_dict_old, strict=False)

                # Get the state of the new layer
                state_dict_new = new_layer.state_dict()

                # Compare keys of both state dicts
                keys_old = set(state_dict_old.keys())
                keys_new = set(k for k in state_dict_new.keys() if not k.startswith("lora_"))
                assert keys_old == keys_new, f"Keys of the state dictionaries don't match (ignoring lora parameters):\n\tExpected Parameters: {keys_old}\n\tNew Parameters (w.o. LoRA): {keys_new}"

                # Replace the original layer with the new layer
                setattr(model, name, new_layer)

            else:
                # Recurse on the child modules
                self.replace_multihead_attention_recursion(module)

    def freeze_parameters_except_lora_and_bias(self):
        """
        Freezes all parameters in the model, except those in LoRA layers, the finetune head, and bias parameters, if specified.
        All lora parameters are identified by having a name that starts with *lora_*.
        All finetune head parameters are identified by having a name that starts with *finetune_head_*.
        """
        for name, param in self.model.named_parameters():
            if ("lora_" in name) or ("finetune_head_" in name) or (self.train_biases and "bias" in name) \
                    or (self.train_embeddings and "embeddings" in name) or (
                    self.train_layer_norms and "LayerNorm" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x, attention_mask=None):
        """The forward method for GLUE, SQuAD, and RACE Benchmarks.
        Calls different split methods for the different tasks.
        """
        if self.task_type == "glue":
            return self.forward_glue(x, attention_mask)
        elif self.task_type == "squad_v1" or self.task_type == "squad":
            return self.forward_squad(x, attention_mask)
        elif self.task_type == "squad_v2":
            return self.forward_squad_v2(x, attention_mask)

    def forward_glue(self, x, attention_mask=None):
        outputs = self.model(x, attention_mask)

        # Take the hidden states output from the base model
        x = outputs.last_hidden_state
        x = x[:, 0, :]  # Take output from [CLS] token

        x = self.finetune_head_norm(x)
        x = self.finetune_head_dropout(x)
        x = self.finetune_head_classifier(x)

        if self.num_classes == 1:  # If it's a regression task
            x = torch.sigmoid(x) * 5  # Scale the output to the range [0, 5] for stsb

        return x

    def forward_squad(self, x, attention_mask=None):
        # Run the input through the base model
        outputs = self.model(x, attention_mask=attention_mask)
        # Take the last hidden state
        x = outputs.last_hidden_state
        # Full sequence is kept as a score

        # Pass the base model output through dropout -> linear layer -> norm
        x = self.finetune_head_norm(x)
        x = self.finetune_head_dropout(x)
        x = self.finetune_head_classifier(x)

        start_logits, end_logits = x.split(1, dim=-1)  # split the two output values

        # Flatten the outputs
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # return the start and end logits
        return start_logits, end_logits

    def forward_squad_v2(self, x, attention_mask=None):
        # Run the input through the base model
        outputs = self.model(x, attention_mask=attention_mask)
        # Take the last hidden state
        x = outputs.last_hidden_state
        # Full sequence is kept as a score

        # Pass the base model output through dropout -> linear layer -> norm
        x = self.finetune_head_norm(x)
        x = self.finetune_head_dropout(x)
        x = self.finetune_head_classifier(x)

        start_logits, end_logits, na_prob_logits = x.split(1, dim=-1)  # split the two output values

        # Flatten the outputs
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        na_prob_logits = na_prob_logits.squeeze(-1)

        # na_prob_logits has to be a single number, only take START token
        na_prob_logits = na_prob_logits[:, 0]

        # return the start and end logits
        return start_logits, end_logits, na_prob_logits

    def save_lora_state_dict(self, lora_filepath: Optional[Union[str, Path]] = None) -> Optional[Dict]:
        """
        Save the trainable parameters of the model into a state dict.
        If a file path is provided, it saves the state dict to that file.
        If no file path is provided, it simply returns the state dict.

        Parameters
        ----------
        lora_filepath : Union[str, Path], optional
            The file path where to save the state dict. Can be a string or a pathlib.Path. If not provided, the function
            simply returns the state dict without saving it to a file.

        Returns
        -------
        Optional[Dict]
            If no file path was provided, it returns the state dict. If a file path was provided, it returns None after saving
            the state dict to the file.
        """
        # Create a state dict of the trainable parameters
        state_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}

        # add addional parameters to state dict
        state_dict['model_id'] = self.model_id
        state_dict['task_type'] = self.task_type
        state_dict['lora_rank'] = self.lora_rank
        state_dict['num_classes'] = self.num_classes

        if lora_filepath is not None:
            # Convert string to pathlib.Path if necessary
            if isinstance(lora_filepath, str):
                lora_filepath = Path(lora_filepath)

            # Save the state dict to the specified file
            torch.save(state_dict, lora_filepath)
        else:
            # Return the state dict if no file path was provided
            return state_dict

    @staticmethod
    def load_lora_state_dict(lora_parameters: Union[str, Path, Dict] = None):
        """
        Load a state dict into the model from a specified file path or a state dict directly.
        This is a staticmethod to be used from the base clase, returning a fully initialized and LoRA loaded model.

        Parameters
        ----------
        lora_parameters : Union[str, Path, Dict]
            Either the file path to the state dict (can be a string or pathlib.Path) or the state dict itself. If a file path
            is provided, the function will load the state dict from the file. If a state dict is provided directly, the function
            will use it as is.

        Returns
        -------
        LoraWrapperRoberta object, initialized and with the LoRA weights loaded.
        """
        # Check if a filepath or state dict was provided
        if lora_parameters is not None:
            # Convert string to pathlib.Path if necessary
            if isinstance(lora_parameters, str):
                lora_parameters = Path(lora_parameters)

            # If the provided object is a Path, load the state dict from file
            if isinstance(lora_parameters, Path):
                state_dict = torch.load(lora_parameters)
            else:
                # If it's not a Path, assume it's a state dict
                state_dict = lora_parameters
        else:
            raise ValueError("No filepath or state dict provided")

        instance = LoraWrapperRoberta(task_type=state_dict['task_type'], num_classes=state_dict['num_classes'],
                                      model_id=state_dict['model_id'], lora_rank=state_dict['lora_rank'])

        # Load the state dict into the model
        instance.load_state_dict(state_dict, strict=False)

        return instance

# Generated test!

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss
from datasets import load_dataset
from transformers import RobertaTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
import numpy as np
from tqdm import tqdm


class GLUEDataset(Dataset):
    def __init__(self, task_name, split, tokenizer, max_length=128):
        """
        Initialize GLUE dataset

        Args:
            task_name: Name of the GLUE task (e.g., 'cola', 'sst2', 'mrpc', etc.)
            split: Dataset split ('train', 'validation', 'test')
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
        """
        self.dataset = load_dataset('glue', task_name)[split]
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.max_length = max_length

        # Map task names to their corresponding label columns and number of labels
        self.task_to_labels = {
            'cola': ('label', 2),
            'sst2': ('label', 2),
            'mrpc': ('label', 2),
            'qqp': ('label', 2),
            'stsb': ('label', 1),  # Regression task
            'mnli': ('label', 3),
            'qnli': ('label', 2),
            'rte': ('label', 2),
            'wnli': ('label', 2)
        }

        self.label_col, self.num_labels = self.task_to_labels[task_name]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Handle different task formats
        if self.task_name in ['cola', 'sst2']:
            text = example['sentence']
            encoding = self.tokenizer(text,
                                      truncation=True,
                                      max_length=self.max_length,
                                      padding='max_length',
                                      return_tensors='pt')
        elif self.task_name in ['mrpc', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']:
            text = example['sentence1']
            text_pair = example['sentence2']
            encoding = self.tokenizer(text, text_pair,
                                      truncation=True,
                                      max_length=self.max_length,
                                      padding='max_length',
                                      return_tensors='pt')
        elif self.task_name == 'stsb':
            text = example['sentence1']
            text_pair = example['sentence2']
            encoding = self.tokenizer(text, text_pair,
                                      truncation=True,
                                      max_length=self.max_length,
                                      padding='max_length',
                                      return_tensors='pt')

        # Convert label to tensor
        label = torch.tensor(example[self.label_col])
        if self.task_name == 'stsb':
            label = label.float()

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }


def compute_metrics(task_name, preds, labels):
    """Compute metrics for GLUE tasks"""
    if task_name == 'cola':
        return {'mcc': matthews_corrcoef(labels, preds)}
    elif task_name in ['sst2', 'mrpc', 'qqp', 'qnli', 'rte', 'wnli']:
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {'accuracy': acc, 'f1': f1}
    elif task_name == 'stsb':
        return {'pearson': np.corrcoef(preds, labels)[0, 1],
                'spearman': scipy.stats.spearmanr(preds, labels)[0]}
    elif task_name == 'mnli':
        return {'accuracy': accuracy_score(labels, preds)}


def train_epoch(model, train_loader, optimizer, criterion, device, task_name):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        if task_name == 'stsb':
            loss = criterion(outputs.squeeze(), labels)
            preds = outputs.squeeze().detach().cpu().numpy()
        else:
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': loss.item()})

    metrics = compute_metrics(task_name, np.array(all_preds), np.array(all_labels))
    metrics['loss'] = total_loss / len(train_loader)
    return metrics


def evaluate(model, eval_loader, criterion, device, task_name):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)

            if task_name == 'stsb':
                loss = criterion(outputs.squeeze(), labels)
                preds = outputs.squeeze().cpu().numpy()
            else:
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

            total_loss += loss.item()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(task_name, np.array(all_preds), np.array(all_labels))
    metrics['loss'] = total_loss / len(eval_loader)
    return metrics


def main():
    # Configuration
    task_name = 'sst2'  # Choose from GLUE tasks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 3
    learning_rate = 1e-4
    max_length = 128

    print(f"Running GLUE task: {task_name}")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Load datasets
    train_dataset = GLUEDataset(task_name, 'train', tokenizer, max_length)
    eval_dataset = GLUEDataset(task_name, 'validation', tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    # Initialize model
    num_labels = train_dataset.num_labels
    model = LoraWrapperRoberta(
        task_type='glue',
        num_classes=num_labels,
        lora_rank=8,
        train_biases=True,
        train_embedding=False,
        train_layer_norms=True
    )
    model.to(device)

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.2f}%")

    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = MSELoss() if task_name == 'stsb' else CrossEntropyLoss()

    # Training loop
    best_metric = float('-inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, task_name)
        print("\nTraining metrics:")
        for k, v in train_metrics.items():
            print(f"{k}: {v:.4f}")

        # Evaluation
        eval_metrics = evaluate(model, eval_loader, criterion, device, task_name)
        print("\nValidation metrics:")
        for k, v in eval_metrics.items():
            print(f"{k}: {v:.4f}")

        # Save best model
        current_metric = eval_metrics['accuracy'] if 'accuracy' in eval_metrics else eval_metrics['mcc']
        if current_metric > best_metric:
            best_metric = current_metric
            model.save_lora_state_dict(f'lora_weights_{task_name}_best.pt')

    print("\nTraining completed!")
    print(f"Best validation metric: {best_metric:.4f}")


if __name__ == "__main__":
    main()
