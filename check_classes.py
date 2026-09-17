from ultralytics import YOLO

model = YOLO("best.pt")

print("Classes:")
for i, name in model.names.items():
    print(i, ":", name)
    