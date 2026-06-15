from ultralytics import YOLO

model = YOLO("best.pt")

print("Task:", model.task)
print("Number of classes:", len(model.names))
print("Class names:")
for class_id, class_name in model.names.items():
    print(f"{class_id}: \"{class_name}\"")