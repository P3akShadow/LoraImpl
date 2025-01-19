import numbers

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


def compute_cider(predictions, references):
    """
    Custom CIDEr implementation
    """
    import nltk
    from collections import Counter
    import numpy as np
    import math
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    def preprocess_text(text):
        return nltk.word_tokenize(text.lower().strip())
    
    def compute_ngrams(words, n):
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    
    def compute_tf(words, n):
        ngrams = compute_ngrams(words, n)
        counter = Counter(ngrams)
        total = sum(counter.values())
        return {gram: count/total for gram, count in counter.items()} if total > 0 else counter
    
    def compute_idf(all_refs, n):
        doc_count = Counter()
        total_docs = len(all_refs)
        
        for refs in all_refs:
            seen_grams = set()
            for ref in refs:
                words = preprocess_text(ref)
                ngrams = compute_ngrams(words, n)
                seen_grams.update(ngrams)
            doc_count.update(seen_grams)
        
        idf_dict = {gram: math.log(total_docs/(count + 1)) for gram, count in doc_count.items()}
        return idf_dict
    
    def compute_cider_score(pred, refs, n, idf):
        pred_words = preprocess_text(pred)
        pred_tf = compute_tf(pred_words, n)
        
        scores = []
        for ref in refs:
            ref_words = preprocess_text(ref)
            ref_tf = compute_tf(ref_words, n)
            
            common_grams = set(pred_tf.keys()) & set(ref_tf.keys())
            
            if len(common_grams) == 0:
                continue
                
            numerator = sum(pred_tf[gram] * ref_tf[gram] * (idf.get(gram, 0) ** 2) for gram in common_grams)
            
            pred_norm = math.sqrt(sum((tf * idf.get(gram, 0) ** 2) ** 2 for gram, tf in pred_tf.items()))
            ref_norm = math.sqrt(sum((tf * idf.get(gram, 0) ** 2) ** 2 for gram, tf in ref_tf.items()))
            
            if pred_norm > 0 and ref_norm > 0:
                score = numerator / (pred_norm * ref_norm)
                scores.append(score)
        
        return max(scores) if scores else 0

    if not predictions or not references or len(predictions) != len(references):
        return 0.0

    n_values = range(1, 5)
    weights = [1/4] * 4
    
    scores = []
    for n in n_values:
        idf = compute_idf(references, n)
        score = np.mean([compute_cider_score(pred, refs, n, idf) 
                        for pred, refs in zip(predictions, references)])
        scores.append(score)
    
    return sum(w * s for w, s in zip(weights, scores))

def evaluate_nlg(model, eval_loader, tokenizer, device):
    model.eval()
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    
    import nltk
    from nltk.translate.nist_score import corpus_nist
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    metrics = dict()
    n_batches = len(eval_loader)
    
    all_predictions = []
    all_references = []
    raw_predictions = []
    raw_references = []
    
    for batch, references in tqdm(eval_loader, desc='Evaluating'):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_metrics = dict()
        
        outputs = model.generate(**batch, pad_token_id=tokenizer.eos_token_id)
        outputs = outputs[:, batch['input_ids'].shape[1]:]
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        batch_metrics |= bleu.compute(predictions=predictions, references=references)
        batch_metrics |= rouge.compute(predictions=predictions, references=references)
        batch_metrics |= meteor.compute(predictions=predictions, references=references)
        
        all_predictions.extend([nltk.word_tokenize(pred.lower()) for pred in predictions])
        all_references.extend([[nltk.word_tokenize(ref.lower()) for ref in refs] for refs in references])
        
        raw_predictions.extend([pred.strip() for pred in predictions])
        raw_references.extend([refs for refs in references])
        
        for key, value in batch_metrics.items():
            if not isinstance(value, numbers.Number):
                continue
            if key not in metrics:
                metrics[key] = value / n_batches
            else:
                metrics[key] += value / n_batches
    
    try:
        nist_score = corpus_nist(all_references, all_predictions, n=4)
        metrics['nist'] = nist_score
    except Exception as e:
        metrics['nist'] = 0.0
    
    try:
        cider_score = compute_cider(raw_predictions, raw_references)
        metrics['cider'] = cider_score
    except Exception:
        metrics['cider'] = 0.0
    
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
    dl = torch.utils.data.DataLoader(ds, collate_fn=cf.validation, batch_size=2)


    mtr = evaluate_nlg(m, dl, t, torch.device('cpu'))
    print(mtr)