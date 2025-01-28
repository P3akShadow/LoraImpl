import torch
from torch import nn
import transformers
from transformers import RobertaTokenizer
import wandb

from loraimpl.data.glue import GLUEDataset
from loraimpl.models.bareFinetuning_roberta import FinetuneWrapperRoberta, FinetuneRoberta
from loraimpl.models.lora_roberta import LoRALinearRoberta
from loraimpl.utils.helper import train_epoch, evaluate_glue, summarize_model

import random as rnd
import numpy as np

import matplotlib.pyplot as plt

def decompose_svg(matrix, comp_rank=32):
    A, S, B = np.linalg.svd(matrix)

    # Truncate to the desired rank
    A_truncated = A[:, :comp_rank]
    S_truncated = np.diag(S[:comp_rank])
    B_truncated = B[:comp_rank, :]

    # Reconstruct compressed matrices A and B
    A = np.dot(A_truncated, S_truncated)  # Shape: (n, comp_rank)
    B = B_truncated  # Shape: (rank, n)
    return A, B

def decomp_chart(matrix, iteration="unspec"):
    A, S, B = np.linalg.svd(matrix)

    eigenvalues = list(S)

    plt.title("Eienvalue Sizes")
    plt.xlabel("eingenvalue number")
    plt.ylabel("eingenvalue")
    plt.plot(range(len(S)), eigenvalues)
    plt.savefig(f"../charts/eignenvalues_example_{iteration}.png")
    plt.close()

    compressed_ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    compressed_distances = []

    print(f"matrix norm: {np.linalg.norm(matrix)}")
    
    for comp_rank in compressed_ranks:
        A, B = decompose_svg(matrix, comp_rank)

        compressed_matrix = np.dot(A, B)
        distance = np.linalg.norm(matrix - compressed_matrix)

        compressed_distances.append(distance)
        print(f"rank: {comp_rank}; distance: {distance}")

    plt.title("Matrix difference")
    plt.xlabel("compression rank")
    plt.ylabel("difference")
    plt.plot(compressed_ranks, compressed_distances)
    plt.savefig(f"../charts/compressed_distances_{iteration}.png")
    plt.close()
        

def main():    
    # Configuration
    num_epochs = 20
    model_name = 'roberta-base'
    model_config = {
        'task_type': 'glue',
        'train_biases': True,
        'train_embedding': False,
        'train_layer_norms': True
    }
    train_dataset_config = {
        'task_name': 'sst2',
        'split': 'train',
        'max_length': 128,
    }
    val_dataset_config = {
        'task_name': 'sst2',
        'split': 'validation',
        'max_length': 128,
    }
    train_loader_config = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True
    }
    val_loader_config = {
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': True
    }
    optimizer_config = {
        'lr': 1e-4,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }
    seed = 42

    # Log configuration to Weights & Biases and run experiment
    config = {
        'model_name': model_name,
        'num_epochs': num_epochs,
        'model_config': model_config,
        'train_dataset_config': train_dataset_config,
        'val_dataset_config': val_dataset_config,
        'train_loader_config': train_loader_config,
        'val_loader_config': val_loader_config,
        'optimizer_config': optimizer_config,
        'seed': seed,
    }

    #config["train_dataset_config"]["task_name"] = "cola"
    #config["val_dataset_config"]["task_name"] = "cola"
    
    wandb.init(project="lora", config=config)
    run_experiment(**config)
    quit()


    #run experiments in loop
    for task in ["stsb", "cola", "sst2"]:
        config["train_dataset_config"]["task_name"] = task
        config["val_dataset_config"]["task_name"] = task

        wandb_name = f"finetune_bare_roberta__{task}_{rnd.randrange(1000)}"
        wandb.init(project="lora", config=config, name=wandb_name)
        run_experiment(**config)
        wandb.finish()
    wandb.init(project="lora", config=config)
    run_experiment(**config)

def replace_multihead_attention_recursion_ft_to_lora(base_model, model):
        """
        Recursively replaces FinetuneRobertaSelfAttention with LoraRobertaSelfAttention in the given model/module.
        If some components are wrapped in another class this function can recursively apply the replacement to
        find all instances of the Attention.

        Parameters
        ----------
        model : nn.Module
            The PyTorch module (or full model) to modify.
        """
        # Model can also be a module if it contains sub-components
        for name, module in model.named_children():

            if isinstance(module, FinetuneRoberta):
                
                # Create a new LoraMultiheadAttention layer
                new_layer = LoRALinearRoberta(rank=32, config=base_model.model_config)

                # Get the state of the original layer
                state_dict_old = module.state_dict()
                
                # Load the state dict to the new layer
                new_layer.load_state_dict(state_dict_old, strict=False)
                
                for key in state_dict_old.keys():
                    if key == "ft_query":
                        query_cpu = state_dict_old[key].cpu().detach().numpy()
                        A,B = decompose_svg(query_cpu)
                        
                        new_layer.lora_query_A = nn.Parameter(torch.from_numpy(A))
                        new_layer.lora_query_B = nn.Parameter(torch.from_numpy(B))
                    if key == "ft_value":
                        value_cpu = state_dict_old[key].cpu().detach().numpy()
                        A,B = decompose_svg(query_cpu)
                        
                        new_layer.lora_value_A = nn.Parameter(torch.from_numpy(A))
                        new_layer.lora_value_B = nn.Parameter(torch.from_numpy(B))


                

                # Get the state of the new layer
                state_dict_new = new_layer.state_dict()

                # Compare keys of both state dicts
                keys_old = set(k for k in state_dict_old.keys() if not k.startswith("ft_"))
                keys_new = set(k for k in state_dict_new.keys() if not k.startswith("lora_"))
                assert keys_old == keys_new, f"Keys of the state dictionaries don't match (ignoring lora parameters):\n\tExpected Parameters: {keys_old}\n\tNew Parameters (w.o. LoRA): {keys_new}"

                # Replace the original layer with the new layer
                setattr(model, name, new_layer)

            else:
                # Recurse on the child modules
                replace_multihead_attention_recursion_ft_to_lora(base_model, module)

def run_experiment(model_name, model_config, train_loader_config, val_loader_config, train_dataset_config, val_dataset_config, num_epochs, optimizer_config, seed=None, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if seed is not None:
        transformers.enable_full_determinism(seed=seed, warn_only=True)

    task_name = train_dataset_config['task_name']

    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Load datasets
    train_dataset = GLUEDataset(tokenizer=tokenizer, **train_dataset_config)
    eval_dataset = GLUEDataset(tokenizer=tokenizer, **val_dataset_config)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_config)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, **val_loader_config)

    # Initialize model
    model = FinetuneWrapperRoberta(num_classes=train_dataset.num_labels, **model_config)
    model.to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    criterion = torch.nn.MSELoss() if task_name == 'stsb' else torch.nn.CrossEntropyLoss()

    summarize_model(model, dataloader=train_loader, device=device)
    print(f"\nTraining for {num_epochs} epochs...")

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
        eval_metrics = evaluate_glue(model, eval_loader, criterion, device, task_name)
        print("\nValidation metrics:")
        for k, v in eval_metrics.items():
            print(f"{k}: {v:.4f}")

        # Save best model
        current_metric = {}
        if 'accuracy' in eval_metrics:
            current_metric = eval_metrics['accuracy']
        elif 'mcc' in eval_metrics:
            current_metric = eval_metrics['mcc']
        elif 'pearson' in eval_metrics:
            current_metric = eval_metrics['pearson']
        if current_metric > best_metric:
            best_metric = current_metric
            model.save_state_dict(f'lora_weights_{task_name}_best.pt')

        # Log metrics to Weights & Biases
        wandb.log({
            'epoch': epoch + 1,
            'loss': train_metrics['loss'],
            'validation_metrics': eval_metrics,
        })
        for name, param in model.model.named_parameters():
            if "ft_" in name:
                print("making charts")
                matrix = param.cpu().detach().numpy()
                decomp_chart(matrix, epoch)
                break




    print("\nTraining completed!")
    print(f"Best validation metric: {best_metric:.4f}")

    


    
    print("replacing full layers with lora")
    replace_multihead_attention_recursion_ft_to_lora(model, model)
    model.to(device)
    summarize_model(model, dataloader=train_loader, device=device)

    
    eval_metrics = evaluate_glue(model, eval_loader, criterion, device, task_name)
    print("\nValidation metrics:")
    for k, v in eval_metrics.items():
        print(f"{k}: {v:.4f}")
    train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, task_name)
    print("\nTraining metrics:")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")
    eval_metrics = evaluate_glue(model, eval_loader, criterion, device, task_name)
    print("\nValidation metrics:")
    for k, v in eval_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    main()
