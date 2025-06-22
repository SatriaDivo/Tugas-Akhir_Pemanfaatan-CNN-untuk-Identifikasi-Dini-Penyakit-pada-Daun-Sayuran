from ultralytics import YOLO
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import torch

# 📌 Bagian 1: Inference dengan model YOLOv8m-cls
model_path = 'runs/classify/klasifikasi_daun/weights/best.pt'
model = YOLO(model_path)

# Daftar gambar uji (pastikan file ada di direktori)
image_paths = [
    'Early-blight-potato.png',
    'early-blight-of-pepper.jpg',
    'tomato-Early-Blight.png'
]

print("🔎 Hasil prediksi YOLOv8m-cls:")
for img_path in image_paths:
    if not os.path.exists(img_path):
        print(f"❌ File tidak ditemukan: {img_path}")
        continue

    result = model(img_path)
    probs = result[0].probs
    pred_label = result[0].names[probs.top1] if hasattr(result[0], 'names') else probs.top1
    print(f"🖼️ {img_path} → Prediksi: {pred_label}, Confidence: {probs.top1conf:.2f}")

# 📌 Bagian 2: Grad-CAM Visualisasi menggunakan ResNet18
print("\n📘 Grad-CAM Visualisasi menggunakan ResNet18 (proxy interpretasi)")

# Load model CNN proxy (pretrained)
cam_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
cam_model.eval()

# Aktifkan gradien pada semua layer agar CAM bisa berjalan
for param in cam_model.parameters():
    param.requires_grad_()

# Inisialisasi CAM extractor
cam_extractor = SmoothGradCAMpp(cam_model)

# Transformasi input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Proses tiap gambar
for img_path in image_paths:
    if not os.path.exists(img_path):
        continue

    img_pil = Image.open(img_path).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0)

    # Jangan gunakan torch.no_grad() agar CAM bisa bekerja
    output = cam_model(input_tensor)
    class_id = output.squeeze(0).argmax().item()

    # Ambil peta aktivasi dari CAM
    activation_map = cam_extractor(class_id, output)

    # Gabungkan heatmap dengan gambar asli
    # Konversi mask ke format gambar (PIL.Image)
    heatmap = activation_map[0].squeeze(0).cpu().numpy()
    heatmap = (heatmap * 255).astype(np.uint8)  # Skala ke 0–255
    heatmap_pil = Image.fromarray(heatmap).resize(img_pil.size).convert("L")
    
    # Overlay heatmap ke gambar asli
    result = overlay_mask(img_pil, heatmap_pil, alpha=0.5)

    # Tampilkan hasil
    plt.imshow(result)
    plt.title(f"Grad-CAM Visual untuk Class ID: {class_id}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
