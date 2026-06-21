import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# الموديلات
# -------------------------
models = ['CNN', 'KNN', 'SVM', 'Logistic']

# -------------------------
# النتائج النهائية لكل موديل (أرقام من evaluation)
# -------------------------
accuracy  = [0.964, 0.604, 0.568, 0.564]
precision = [0.96, 0.61, 0.57, 0.56]
recall    = [0.96, 0.60, 0.57, 0.56]
f1_score  = [0.96, 0.60, 0.57, 0.56]

metrics = [accuracy, precision, recall, f1_score]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']

# -------------------------
# رسم كل مقياس
# -------------------------
x = np.arange(len(models))

for i, m in enumerate(metrics):
    plt.figure(figsize=(8,6))
    plt.bar(x, m, color='skyblue')
    plt.xticks(x, models)
    plt.ylim(0,1)
    for j, val in enumerate(m):
        plt.text(j, val+0.01, f"{val:.2f}", ha='center', fontsize=10)
    plt.title(f"{metric_names[i]} Comparison Across Models", fontsize=14)
    plt.ylabel(metric_names[i], fontsize=12)
    plt.xlabel("Models", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{metric_names[i]}_comparison.png")
    plt.show()
