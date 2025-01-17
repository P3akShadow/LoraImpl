# Basic Imports
from pathlib import Path
from typing import Union, Dict, Optional, Tuple

import torch
from torch import nn
import math

# Basic Model Imports
from transformers import RobertaTokenizer, RobertaModel

# Import Attentions to adjust
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

class WrapperRoberta(nn.Module):
    def __init__(self, task_type, num_classes: int = None, dropout_rate=0.1, model_id: str = "roberta-base",
                 train_biases: bool = True, train_embedding: bool = False, train_layer_norms: bool = True, hidden_dim=100):
        """
        Initializes a WrapperRoberta instance, which is a wrapper around the RoBERTa model incorporating
        to retrain the model for different NLP tasks such as GLUE benchmarks
        and SQuAD.

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
        train_biases : bool, default True
            Flag indicating whether to update bias parameters during training.
        train_embedding : bool, default False
            Flag indicating whether to update embedding layer weights during training.
        train_layer_norms : bool, default True
            Flag indicating whether to update the layer norms during training. Usually this is a good idea.

        Examples
        --------
        To initialize a model for the 'glue' task type:

            model = FinetuneWrapperRoberta(task_type='squad_v1')
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

        self.train_biases = train_biases
        self.train_embeddings = train_embedding
        self.train_layer_norms = train_layer_norms

        # 2. Add the layer for the benchmark tasks
        # Get the output size of the base model & save other parameters
        d_model = self.model_config.hidden_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.task_type = task_type.lower()

        # Define the additional norm, linear layer, hidden layer, and dropout
        self.finetune_head_norm = nn.LayerNorm(d_model)
        self.finetune_head_dropout = nn.Dropout(dropout_rate)
        self.finetune_head_hidden = nn.Linear(d_model, hidden_dim)
        self.finetune_head_hidden_dropout = nn.Dropout(dropout_rate)
        self.finetune_head_classifier = nn.Linear(hidden_dim, num_classes)

        # 3. set up the model for training in Benchmark task:
        self.freeze_parameters_except_finetune()

    def freeze_parameters_except_finetune(self):
        """
        Freezes all parameters in the model, except those the finetune head,
        All finetune head parameters are identified by having a name that starts with *finetune_head_*.
        """
        for name, param in self.model.named_parameters():
            if ("finetune_head_" in name):
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
        x = self.finetune_head_hidden(x)
        x = self.finetune_head_hidden_dropout(x)
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

        # Pass the base model output through dropout -> linear layer -> hidden layer -> norm
        x = self.finetune_head_norm(x)
        x = self.finetune_head_dropout(x)
        x = self.finetune_head_hidden(x)
        x = self.finetune_head_hidden_dropout(x)
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

        # Pass the base model output through dropout -> linear layer -> hidden layer -> norm
        x = self.finetune_head_norm(x)
        x = self.finetune_head_dropout(x)
        x = self.finetune_head_hidden(x)
        x = self.finetune_head_hidden_dropout(x)
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

    def save_state_dict(self, filepath: Optional[Union[str, Path]] = None) -> Optional[Dict]:
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
        state_dict['num_classes'] = self.num_classes

        if filepath is not None:
            # Convert string to pathlib.Path if necessary
            if isinstance(filepath, str):
                lora_filepath = Path(filepath)

            # Save the state dict to the specified file
            torch.save(state_dict, lora_filepath)
        else:
            # Return the state dict if no file path was provided
            return state_dict

    @staticmethod
    def load_state_dict(parameters: Union[str, Path, Dict] = None):
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
        if parameters is not None:
            # Convert string to pathlib.Path if necessary
            if isinstance(parameters, str):
                parameters = Path(parameters)

            # If the provided object is a Path, load the state dict from file
            if isinstance(parameters, Path):
                state_dict = torch.load(parameters)
            else:
                # If it's not a Path, assume it's a state dict
                state_dict = parameters
        else:
            raise ValueError("No filepath or state dict provided")

        instance = WrapperRoberta(task_type=state_dict['task_type'], num_classes=state_dict['num_classes'],
                                      model_id=state_dict['model_id'])

        # Load the state dict into the model
        instance.load_state_dict(state_dict, strict=False)

        return instance






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



