import os

preprocessed_dataset_path = 'preprocessed_dataset'
train_dir = os.path.join(preprocessed_dataset_path, 'train')
val_dir = os.path.join(preprocessed_dataset_path, 'val')
config_file_name = 'plant_disease_config.yaml'
config_file_path = os.path.join(preprocessed_dataset_path, config_file_name)

abs_train_path = os.path.abspath(train_dir)
abs_val_path = os.path.abspath(val_dir)

# Ambil daftar kelas
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# Validasi kelas
if not class_names:
    raise ValueError("❌ Tidak ditemukan kelas pada folder train!")

print(f"✅ Ditemukan {len(class_names)} kelas: {class_names}")

# Tulis file YAML
with open(config_file_path, 'w') as f:
    f.write(f"train: {abs_train_path}\n")
    f.write(f"val: {abs_val_path}\n")
    f.write("names:\n")
    for name in class_names:
        f.write(f"- {name}\n")

print(f"✅ File konfigurasi YOLOv8 berhasil dibuat: {config_file_path}")
