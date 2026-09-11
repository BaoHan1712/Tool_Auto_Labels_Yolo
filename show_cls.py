from ultralytics import YOLO

# Load model
model = YOLO("2class.pt")

# In ra toàn bộ dictionary {id: 'tên_class'}
print(model.names)

# In theo từng dòng cho dễ nhìn:
for class_id, class_name in model.names.items():
    print(f"ID {class_id}: {class_name}")