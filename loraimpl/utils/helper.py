import torch
import numpy as np
import scipy
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
import evaluate
import torchinfo


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


def evaluate_nlg(model, eval_loader, tokenizer, device):
    model.eval()

    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')

    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # we won't heavily tune generation here
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=True,
                top_p=0.85,
                temperature=0.6,
                num_beams=5,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                early_stopping=True,
            )

            for generated_seq in outputs:
                decoded = tokenizer.decode(generated_seq, skip_special_tokens=True)
                all_preds.append(decoded.strip())

            for lbl in batch['labels']:
                # Convert the -100 to pad
                lbl = torch.where(lbl == -100, tokenizer.pad_token_id, lbl)
                decoded = tokenizer.decode(lbl, skip_special_tokens=True)
                all_refs.append(decoded.strip())

    # Show an example
    if len(all_preds) > 0:
        print("\nExample generation:")
        print("Pred:", all_preds[0])
        print("Ref: ", all_refs[0])

    # Evaluate
    results = {}
    results["bleu"] = bleu.compute(
        predictions=all_preds,
        references=[[r] for r in all_refs]
    )["bleu"]

    r_scores = rouge.compute(predictions=all_preds, references=all_refs)
    results["rouge1"] = r_scores["rouge1"]
    results["rouge2"] = r_scores["rouge2"]
    results["rougeL"] = r_scores["rougeL"]

    results["meteor"] = meteor.compute(
        predictions=all_preds,
        references=all_refs
    )["meteor"]

    return results


def summarize_model(model, dataloader=None):
    """Describe the model"""
    if dataloader is None:
        torchinfo.summary(model)
        return
    example_input = next(iter(dataloader))['input_ids']
    torchinfo.summary(model, example_input.size())
