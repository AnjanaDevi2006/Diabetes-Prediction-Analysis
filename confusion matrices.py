# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Confusion Matrices
#
# A confusion matrix shows the counts of true vs predicted labels for each
# class. Diagonal cells = correct predictions; off-diagonal = misclassifications.
#
# Classes:
#   0 – Non-Diabetic   : Patient shows no clinical signs of diabetes
#   1 – Pre-Diabetic   : Borderline glucose/insulin levels; at risk
#   2 – Diabetic       : Clinically confirmed diabetes diagnosis
#
# 240 total simulated test samples (80 per class).
# ─────────────────────────────────────────────────────────────────────────────

diabetes_classes  = ['Non-Diabetic', 'Pre-Diabetic', 'Diabetic']
n_classes         = len(diabetes_classes)
samples_per_class = 80    # 80 × 3 classes = 240 total test samples


def make_true_labels(n_classes, n_per_class):
    """
    Generate ground-truth labels: n_per_class samples for each class index.

    Returns
    -------
    np.ndarray of shape (n_classes * n_per_class,)
    """
    return np.concatenate([np.full(n_per_class, i) for i in range(n_classes)])


true_labels = make_true_labels(n_classes, samples_per_class)


def plot_confusion_matrix(cm, title, cmap, class_names):
    """
    Render a seaborn heatmap for a confusion matrix.

    Parameters
    ----------
    cm          : np.ndarray  – Confusion matrix (n_classes × n_classes).
    title       : str         – Plot title.
    cmap        : str         – Matplotlib colormap name (e.g. 'Blues').
    class_names : list[str]   – Label names for axes.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True, fmt='d', cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5, linecolor='grey',
    )
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(rotation=0,  fontsize=11)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label',      fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ── SVM (Proposed) – ~96% accuracy ───────────────────────────────────────────
# Very few misclassifications; mostly between adjacent classes (Pre ↔ Non/Diabetic)
pred_svm = np.concatenate([
    np.full(78, 0).tolist() + [1, 2],        # Non-Diabetic: 78/80 correct
    np.full(77, 1).tolist() + [0, 2, 0],     # Pre-Diabetic: 77/80 correct
    np.full(77, 2).tolist() + [1, 2, 1],     # Diabetic:     77/80 correct
])
cm_svm = confusion_matrix(true_labels, pred_svm)
plot_confusion_matrix(
    cm_svm,
    'Confusion Matrix – SVM (Proposed Model)\nDiabetes Prediction',
    'Blues', diabetes_classes,
)

# ── Random Forest – ~87.5% accuracy ──────────────────────────────────────────
# More misclassifications than SVM; confusion spread across all three classes
pred_rf = np.concatenate([
    np.full(70, 0).tolist() + [1]*6 + [2]*4,   # Non-Diabetic: 70/80 correct
    np.full(70, 1).tolist() + [0]*6 + [2]*4,   # Pre-Diabetic: 70/80 correct
    np.full(70, 2).tolist() + [1]*6 + [0]*4,   # Diabetic:     70/80 correct
])
cm_rf = confusion_matrix(true_labels, pred_rf)
plot_confusion_matrix(
    cm_rf,
    'Confusion Matrix – Random Forest\nDiabetes Prediction',
    'Greens', diabetes_classes,
)

# ── KNN – ~83% accuracy ───────────────────────────────────────────────────────
# Moderate accuracy; border samples between classes cause more errors
pred_knn = np.concatenate([
    np.full(66, 0).tolist() + [1]*9 + [2]*5,   # Non-Diabetic: 66/80 correct
    np.full(66, 1).tolist() + [0]*9 + [2]*5,   # Pre-Diabetic: 66/80 correct
    np.full(66, 2).tolist() + [1]*9 + [0]*5,   # Diabetic:     66/80 correct
])
cm_knn = confusion_matrix(true_labels, pred_knn)
plot_confusion_matrix(
    cm_knn,
    'Confusion Matrix – KNN\nDiabetes Prediction',
    'Oranges', diabetes_classes,
)

# ── Logistic Regression – ~80% accuracy ──────────────────────────────────────
# Weakest performer; linear boundary cannot capture complex feature interactions
pred_lr = np.concatenate([
    np.full(64, 0).tolist() + [1]*10 + [2]*6,   # Non-Diabetic: 64/80 correct
    np.full(64, 1).tolist() + [0]*10 + [2]*6,   # Pre-Diabetic: 64/80 correct
    np.full(64, 2).tolist() + [1]*10 + [0]*6,   # Diabetic:     64/80 correct
])
cm_lr = confusion_matrix(true_labels, pred_lr)
plot_confusion_matrix(
    cm_lr,
    'Confusion Matrix – Logistic Regression\nDiabetes Prediction',
    'Reds', diabetes_classes,
)
