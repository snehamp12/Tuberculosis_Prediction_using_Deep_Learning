import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import numpy as np

# Load VGG16 metrics from JSON file
with open("vgg16_metrics.json", "r") as f:
    metrics = json.load(f)

# Unpack data
train_loss = metrics["train_loss"]
val_loss = metrics["val_loss"]
train_acc = metrics["train_acc"]
val_acc = metrics["val_acc"]
predictions = metrics["predictions"]
targets = metrics["targets"]

class_names = ["Healthy", "Tuberculosis", "Other Lung Disease"]
n_classes = len(class_names)

# Plot 1: Accuracy
plt.figure(figsize=(8, 5))
plt.plot(train_acc, label="Train Accuracy", marker="o")
plt.plot(val_acc, label="Val Accuracy", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vgg16_accuracy_curve.png")
plt.show()

# Plot 2: Loss
plt.figure(figsize=(8, 5))
plt.plot(train_loss, label="Train Loss", marker="o")
plt.plot(val_loss, label="Val Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vgg16_loss_curve.png")
plt.show()

# Plot 3: Confusion Matrix
cm = confusion_matrix(targets, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("vgg16_confusion_matrix.png")
plt.show()

# Plot 4: Classification Report
report = classification_report(targets, predictions, target_names=class_names, output_dict=True)
report_text = classification_report(targets, predictions, target_names=class_names)
print("\nClassification Report:\n", report_text)

# Plot 5: ROC Curve (One-vs-Rest)
y_true = label_binarize(targets, classes=[0, 1, 2])
y_pred = label_binarize(predictions, classes=[0, 1, 2])
fpr, tpr, roc_auc = {}, {}, {}

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vgg16_roc_curve.png")
plt.show()

# Plot 6: Precision-Recall Curve
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
    ap = average_precision_score(y_true[:, i], y_pred[:, i])
    plt.plot(recall, precision, label=f"{class_names[i]} (AP = {ap:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (One-vs-Rest)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vgg16_precision_recall_curve.png")
plt.show()

# Plot 7: Class-wise Accuracy Bar Chart
correct = np.array(predictions) == np.array(targets)
per_class_acc = []
for i in range(n_classes):
    idx = np.array(targets) == i
    acc = 100 * np.sum(correct[idx]) / np.sum(idx)
    per_class_acc.append(acc)

plt.figure(figsize=(7, 5))
sns.barplot(x=class_names, y=per_class_acc)
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.title("Class-wise Accuracy Comparison")
plt.tight_layout()
plt.savefig("vgg16_classwise_accuracy.png")
plt.show()
