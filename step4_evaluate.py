from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from PIL import Image
import pandas as pd

# ------------------------------
# 🔍 Load model dan evaluasi
# ------------------------------
model_path = 'runs/classify/klasifikasi_daun/weights/best.pt'
model = YOLO(model_path)

print(f"\n🔍 Evaluating model: {model_path}")
results = model.val()

# ------------------------------
# 📊 Ringkasan hasil evaluasi
# ------------------------------
print("\n📊 Evaluasi:")
print(f"- Top-1 Accuracy : {results.top1:.4f}")
print(f"- Top-5 Accuracy : {results.top5:.4f}")
print(f"- Fitness Score  : {results.fitness:.4f}")
print(f"- Dataset Path   : {results.save_dir}\n")

# ------------------------------
# 📌 Confusion Matrix
# ------------------------------
if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
    cm = results.confusion_matrix.matrix.astype(int)
    labels = results.names if hasattr(results, 'names') else [f"Class {i}" for i in range(len(cm))]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='YlGnBu',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # --------------------------------------
    # 📌 Bar Chart Akurasi Per Kelas
    # --------------------------------------
    correct = np.diag(cm)
    total = cm.sum(axis=1)
    class_accuracy = np.where(total != 0, correct / total, 0)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=labels, y=class_accuracy * 100, palette='crest')
    plt.ylabel("Akurasi (%)")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.title("Akurasi Per Kelas (Sehat vs Penyakit)")
    plt.tight_layout()
    plt.show()

# ------------------------------
# 📈 Grafik Pelatihan (Loss & Akurasi)
# ------------------------------
train_log_path = "runs/classify/klasifikasi_daun/results.csv"
if os.path.exists(train_log_path):
    df = pd.read_csv(train_log_path)
    df.columns = df.columns.str.strip()

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train/loss'], label='Train Loss', color='blue')
    plt.plot(df['epoch'], df['val/loss'], label='Val Loss', color='orange')
    plt.plot(df['epoch'], df['metrics/accuracy_top1'], label='Accuracy Top-1', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.title("Grafik Pelatihan Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ File log pelatihan (results.csv) tidak ditemukan.")
