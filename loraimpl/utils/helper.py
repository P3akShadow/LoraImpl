import numbers

import torch
import scipy
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
import evaluate
import torchinfo
import nltk
from collections import Counter
import numpy as np
import math

from loraimpl.utils.metric import CIDEr


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


def evaluate_glue(model, eval_loader, criterion, device, task_name):
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


def evaluate_nlg(model, eval_loader, tokenizer, device, inference_cfg):
    model.eval()
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    nist = evaluate.load('nist_mt')
    cider = CIDEr()  # Morgans custom CIDEr implementation
    metrics = dict()
    n_batches = len(eval_loader)
    
    for batch, references in (pbar := tqdm(eval_loader, desc='Evaluating')):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_metrics = dict()
        
        outputs = model.generate(**batch, pad_token_id=tokenizer.eos_token_id, **inference_cfg)
        outputs = outputs[:, batch['input_ids'].shape[1]:]
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        batch_metrics |= bleu.compute(predictions=predictions, references=references)
        batch_metrics |= rouge.compute(predictions=predictions, references=references)
        batch_metrics |= meteor.compute(predictions=predictions, references=references)
        batch_metrics |= nist.compute(predictions=predictions, references=references)
        batch_metrics |= cider.compute(predictions=predictions, references=references)

        for key, value in batch_metrics.items():
            if not isinstance(value, numbers.Number):
                continue
            if key not in metrics:
                metrics[key] = value / n_batches
            else:
                metrics[key] += value / n_batches
        pbar.set_description(f'Evaluating: {metrics["bleu"]:0.4f} BLEU')
    
    return metrics
    

def summarize_model(model, dataloader=None, device=None, depth=7):
    """Describe the model"""
    if dataloader is None:
        torchinfo.summary(model, depth=depth)
        return
    example_input = next(iter(dataloader))
    torchinfo.summary(
        model,
        device=device,
        depth=depth,
        **example_input
    )


# debugging and testing
if __name__ == '__main__':
    from transformers import GPT2TokenizerFast
    from loraimpl.models.lora_gpt2 import GPT2LMHeadModelLora
    from datasets import load_dataset
    from loraimpl.data.nlg import CollateFunction

    m = GPT2LMHeadModelLora.from_pretrained("gpt2")

    t = GPT2TokenizerFast.from_pretrained("gpt2", padding_side='left')
    t.padding_side = "left"

    ds = load_dataset('GEM/e2e_nlg', split='validation')
    cf = CollateFunction(t, torch.device('cpu'))
    dl = torch.utils.data.DataLoader(ds, collate_fn=cf.evaluation, batch_size=2)


    mtr = evaluate_nlg(m, dl, t, torch.device('cpu'))
    print(mtr)