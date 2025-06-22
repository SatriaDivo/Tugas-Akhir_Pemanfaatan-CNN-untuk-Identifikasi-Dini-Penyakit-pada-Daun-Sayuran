import os
import shutil
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Path dataset asli dan hasil preprocessing
original_dataset = 'dataset'
preprocessed_dataset = 'preprocessed_dataset'
img_size = 224

# Cek kelas dan jumlah gambar per kelas
print("📦 Jumlah gambar per kelas:")
for class_name in os.listdir(original_dataset):
    class_path = os.path.join(original_dataset, class_name)
    count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"- {class_name}: {count} gambar")

# Buat struktur folder tujuan
for phase in ['train', 'val']:
    for class_name in os.listdir(original_dataset):
        os.makedirs(os.path.join(preprocessed_dataset, phase, class_name), exist_ok=True)

# Fungsi resize + pengecekan gambar
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Gagal membaca gambar: {image_path}")
        return None
    img = cv2.resize(img, (img_size, img_size))
    return img

# Proses split dan simpan
for class_name in os.listdir(original_dataset):
    class_path = os.path.join(original_dataset, class_name)
    image_paths = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    train_imgs, val_imgs = train_test_split(image_paths, test_size=0.3, random_state=42)

    for phase, img_list in zip(['train', 'val'], [train_imgs, val_imgs]):
        for img_path in tqdm(img_list, desc=f"{phase} - {class_name}"):
            img = preprocess_image(img_path)
            if img is not None:
                save_path = os.path.join(preprocessed_dataset, phase, class_name, os.path.basename(img_path))
                cv2.imwrite(save_path, img)

print("✅ Preprocessing & split selesai. Dataset disimpan di folder:", preprocessed_dataset)
