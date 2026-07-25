# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Epoch vs Metric Line Plots
#
# Visualises how each performance metric evolves over training epochs for all
# four models. Each model is assigned a unique line style, colour, and marker
# to distinguish trends clearly.
# ─────────────────────────────────────────────────────────────────────────────

LINE_STYLES = ['--', ':', '-.', '-']   # Dashed, dotted, dash-dot, solid
COLORS      = ['g',  'b',  'r',  'c'] # Green, Blue, Red, Cyan
MARKERS     = ['o',  's',  '^',  'D'] # Circle, Square, Triangle, Diamond

# Maps metric keys to their corresponding plot title
METRIC_CONFIG = {
    'accuracy':  'Accuracy vs Epochs – Diabetes Prediction',
    'precision': 'Precision vs Epochs – Diabetes Prediction',
    'recall':    'Recall vs Epochs – Diabetes Prediction',
    'f1_score':  'F1-Score vs Epochs – Diabetes Prediction',
    'loss':      'Loss vs Epochs – Diabetes Prediction',
}


def plot_metric(metric_key, title, epochs, results, save_path=None):
    """
    Plot a single performance metric across training epochs for all models.

    Parameters
    ----------
    metric_key : str
        Key into the results dict (e.g. 'accuracy', 'loss').
    title      : str
        Title to display on the plot.
    epochs     : list[int]
        Epoch checkpoints (e.g. [10, 20, ..., 100]).
    results    : dict
        Nested dict of model_name → {metric_key → np.array}.
    save_path  : str or None
        If provided, saves the figure to this path.
    """
    plt.figure(figsize=(8, 5))

    for i, (model_name, metrics) in enumerate(results.items()):
        plt.plot(
            epochs,
            metrics[metric_key],
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linewidth=2,
            markersize=7,
            label=model_name,
        )

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(metric_key.replace('_', ' ').title(), fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")

    plt.show()


print("=" * 60)
print("  DIABETES PREDICTION – EPOCH vs METRIC PLOTS")
print("=" * 60)

for metric_key, title in METRIC_CONFIG.items():
    plot_metric(metric_key, title, epochs, results, save_path=None)
