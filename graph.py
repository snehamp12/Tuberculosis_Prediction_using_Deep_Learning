import json
import matplotlib.pyplot as plt

# Use a valid style
plt.style.use('seaborn-v0_8-darkgrid')

# Load ResNet16 metrics
with open("resnet16_metrics.json", "r") as f:
    resnet16_metrics = json.load(f)

# Load VGG16 metrics
with open("vgg16_metrics.json", "r") as f:
    vgg16_metrics = json.load(f)

# Only use first 5 epochs from ResNet
resnet_train_loss = resnet16_metrics["train_loss"][:5]
resnet_val_loss = resnet16_metrics["val_loss"][:5]
resnet_train_acc = resnet16_metrics["train_acc"][:5]
resnet_val_acc = resnet16_metrics["val_acc"][:5]
resnet_times = resnet16_metrics["epoch_times"][:5]

# VGG16 (all 5 epochs)
vgg_train_loss = vgg16_metrics["train_loss"]
vgg_val_loss = vgg16_metrics["val_loss"]
vgg_train_acc = vgg16_metrics["train_acc"]
vgg_val_acc = vgg16_metrics["val_acc"]
vgg_times = vgg16_metrics["epoch_times"]

epochs = list(range(1, 6))

# Plot: Training Accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, resnet_train_acc, label="ResNet16 Train Acc", marker='o')
plt.plot(epochs, vgg_train_acc, label="VGG16 Train Acc", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, resnet_val_acc, label="ResNet16 Val Acc", marker='o')
plt.plot(epochs, vgg_val_acc, label="VGG16 Val Acc", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Training Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, resnet_train_loss, label="ResNet16 Train Loss", marker='o')
plt.plot(epochs, vgg_train_loss, label="VGG16 Train Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, resnet_val_loss, label="ResNet16 Val Loss", marker='o')
plt.plot(epochs, vgg_val_loss, label="VGG16 Val Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Epoch Time Comparison
plt.figure(figsize=(10, 6))
plt.plot(epochs, resnet_times, label="ResNet16 Epoch Time", marker='o')
plt.plot(epochs, vgg_times, label="VGG16 Epoch Time", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
plt.title("Epoch Time Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
