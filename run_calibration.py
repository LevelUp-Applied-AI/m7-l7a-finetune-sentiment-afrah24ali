import json
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from manual_eval import manual_predict, compute_classification_report_from_arrays
from calibration import reliability_diagram, expected_calibration_error, plot_reliability


MODEL_DIR = "model"
DATA_PATH = "data/app_reviews_eval.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    df = pd.read_csv(DATA_PATH)

    texts = df[TEXT_COLUMN].astype(str).tolist()
    y_true = df[LABEL_COLUMN].to_numpy()

    y_pred, probs = manual_predict(model, tokenizer, texts, batch_size=8)

    report = compute_classification_report_from_arrays(y_true, y_pred)
    print(json.dumps(report, indent=2))

    centers, accs, counts = reliability_diagram(probs, y_true, n_bins=10)
    ece = expected_calibration_error(probs, y_true, n_bins=10)

    print("\nReliability buckets:")
    for center, acc, count in zip(centers, accs, counts):
        print(f"center={center:.2f}, accuracy={acc:.3f}, count={count}")

    print(f"\nECE: {ece:.4f}")

    plot_reliability(
        centers,
        accs,
        counts,
        "figures/reliability-diagram.png"
    )


if __name__ == "__main__":
    main()