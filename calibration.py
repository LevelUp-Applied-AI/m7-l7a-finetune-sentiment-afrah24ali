import numpy as np


def reliability_diagram(probs, y_true, n_bins=10):

    y_true = np.array(y_true)

    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    edges = np.linspace(0, 1, n_bins + 1)

    centers = (edges[:-1] + edges[1:]) / 2

    accs = []
    counts = []

    for i in range(n_bins):

        low = edges[i]
        high = edges[i + 1]

        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        count = np.sum(mask)
        counts.append(count)

        if count > 0:
            acc = np.mean(preds[mask] == y_true[mask])
        else:
            acc = 0.0

        accs.append(acc)

    return np.array(centers), np.array(accs), np.array(counts)


def expected_calibration_error(probs, y_true, n_bins=10):

    y_true = np.array(y_true)

    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    edges = np.linspace(0, 1, n_bins + 1)

    N = len(y_true)

    ece = 0.0

    for i in range(n_bins):

        low = edges[i]
        high = edges[i + 1]

        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        count = np.sum(mask)

        if count == 0:
            continue

        acc = np.mean(preds[mask] == y_true[mask])
        conf = np.mean(confidences[mask])

        ece += (count / N) * abs(acc - conf)

    return float(ece)


def plot_reliability(centers, accs, counts, output_path):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))

    plt.bar(centers, accs, width=0.08, edgecolor="black", alpha=0.7)

    plt.plot([0, 1], [0, 1], "--", color="gray")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")

    plt.title("Reliability Diagram")

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)

    plt.close()