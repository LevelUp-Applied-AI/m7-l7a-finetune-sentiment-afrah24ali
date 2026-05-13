import numpy as np
import torch


def manual_predict(model, tokenizer, texts, batch_size=8):

    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_probs = []

    with torch.no_grad():

        for i in range(0, len(texts), batch_size):

            batch = texts[i:i+batch_size]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)

            probs = torch.softmax(outputs.logits, dim=-1)

            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def compute_classification_report_from_arrays(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.sum(y_true == y_pred) / len(y_true)

    num_classes = max(np.max(y_true), np.max(y_pred)) + 1

    per_class = {}
    f1s = []

    for c in range(num_classes):

        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        per_class[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

        f1s.append(f1)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class
    }