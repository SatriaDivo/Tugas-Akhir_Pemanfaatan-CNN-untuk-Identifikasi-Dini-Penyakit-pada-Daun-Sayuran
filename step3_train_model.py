from ultralytics import YOLO
import os

model = YOLO('yolov8m-cls.pt')

# Path to the directory containing the dataset config file
dataset_root_dir = os.path.abspath('preprocessed_dataset')
print("✅ Path ke root dataset:", dataset_root_dir)

output_name = "klasifikasi_daun"

model.train(
    data=dataset_root_dir,
    epochs=50,
    imgsz=224,
    batch=32,
    name=output_name,
    augment=True,
)

print(f"✅ Training selesai! Model tersimpan di: runs/classify/{output_name}/weights/best.pt")
